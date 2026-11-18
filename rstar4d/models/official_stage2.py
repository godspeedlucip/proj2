"""Stage II backbone from official Training_Tetris2.ipynb (cells 8-11).
Authors: Ziheng Deng and Jun Zhao. See provenance.json for source SHA256.
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _encode(x, down, first, second):
    x = down(x)
    return second(first(x)) + x


def _decode(x, skip, up, first, second):
    x = up(x)
    return second(first(torch.cat((x,skip),dim=1))) + x

class convwithactivation(nn.Module):
    # Spatial downsampling layer
    def __init__(self,in_ch,out_ch,kernel_size=[3,3,3],padding=[1,1,1],stride=[1,1,1],padding_mode='zeros'):
        super(convwithactivation,self).__init__()
        # XY downsamapling
        self.conv1=nn.Conv3d(in_ch,out_ch,[1,kernel_size[1],kernel_size[2]],[1,stride[1],stride[2]],[0,padding[1],padding[2]],padding_mode=padding_mode)
        # Z downsampling
        self.conv2=nn.Conv3d(out_ch,out_ch,[1,kernel_size[0],1],[1,stride[0],1],[0,padding[0],0],padding_mode='replicate')
        self.lrelu=nn.LeakyReLU(0.2, inplace=True)
        self.stride=stride
        self.out_ch=out_ch
        
    def forward(self,x):
        [B,C,T,Z,Y,X]=x.shape
        x = x.view(B,C,T*Z,Y,X)
        x = self.conv1(x)
        x = x.view(B,self.out_ch,T,Z,-1)
        x = self.conv2(x)
        x = self.lrelu(x)
        x = x.view(B,self.out_ch,T,Z//self.stride[0],Y//self.stride[1],X//self.stride[2])
        return x

class convwithactivation4DRSTAR(nn.Module):
    # separable 4D convolution
    def __init__(self,in_ch,out_ch,kernel_size=[3,3,3,3],padding=[1,1,1,1],stride=[1,1,1,1],ifrelu=1):
        super(convwithactivation4DRSTAR,self).__init__()
        # XY conv，TZ*Y*X
        self.conv1=nn.Conv3d(int(1*in_ch),int(1*out_ch),[1,kernel_size[2],kernel_size[3]],[1,stride[2],stride[3]],[0,padding[2],padding[3]])
        # Z conv，T*Z*YX
        self.conv2=nn.Conv3d(int(1*in_ch),int(1*out_ch),[1,kernel_size[1],1],[1,stride[1],1],[0,padding[1],0],padding_mode='replicate')
        # T conv，T*Z*YX
        self.conv3=nn.Conv3d(int(1*in_ch),int(1*out_ch),[kernel_size[0],1,1],[stride[0],1,1],[padding[0],0,0],padding_mode='circular')

        # XY conv，TZ*Y*X
        self.conv1_2=nn.Conv3d(int(1*out_ch),int(1*out_ch),[1,kernel_size[2],kernel_size[3]],[1,stride[2],stride[3]],[0,padding[2],padding[3]])
        # Z conv，T*Z*YX
        self.conv2_2=nn.Conv3d(int(1*out_ch),int(1*out_ch),[1,kernel_size[1],1],[1,stride[1],1],[0,padding[1],0],padding_mode='replicate')
        # T conv，T*Z*YX
        self.conv3_2=nn.Conv3d(int(1*out_ch),int(1*out_ch),[kernel_size[0],1,1],[stride[0],1,1],[padding[0],0,0],padding_mode='circular')

        self.lrelu1=nn.LeakyReLU(0.2, inplace=True)
        self.in_ch=in_ch
        self.out_ch=out_ch
        self.ifrelu=ifrelu
        
    def forward(self,x):
        [B,C,T,Z,Y,X]=x.shape
        x = self.conv1(x.view(B,C,T*Z,Y,X)).view(B,self.out_ch,T,Z,-1)+self.conv2(x.view(B,C,T,Z,-1))+self.conv3(x.view(B,C,T,Z,-1))
        x = self.conv1_2(x.view(B,self.out_ch,T*Z,Y,X)).view(B,self.out_ch,T,Z,-1)+self.conv2_2(x)+self.conv3_2(x)

        if(self.ifrelu==1):
            x = self.lrelu1(x)
        x = x.view(B,self.out_ch,T,Z,Y,X)
        return x

class upconvwithactivation(nn.Module):
    # Spatial upsampling
    def __init__(self,in_ch,out_ch,kernel_size=[3,3,3,3],padding=[1,1,1,1],stride=[1,1,1,1],scale_factor=[1,2,2,2],ifrelu=1):
        super(upconvwithactivation,self).__init__()
        self.scale_factor=scale_factor
        self.conv=convwithactivation4DRSTAR(in_ch,out_ch,kernel_size,padding,stride,ifrelu)
    def forward(self,x):
        [B,C,T,Z,Y,X]=x.shape
        x = x.view(B,C*T,Z,Y,X)
        x = F.interpolate(x,scale_factor=self.scale_factor[1:],mode='nearest')
        x = x.view(B,C,T,Z*self.scale_factor[1],Y*self.scale_factor[2],X*self.scale_factor[3])
        x = self.conv(x)
        return x

class myUNet(nn.Module):
    def __init__(self):
        super(myUNet,self).__init__()
        cnum=16
        self.conv1=convwithactivation(2,cnum,kernel_size=[3,3,3],padding=[1,1,1],stride=[2,1,1],padding_mode='zeros')
        self.conv1_2=convwithactivation4DRSTAR(cnum,cnum,kernel_size=[3,3,3,3],padding=[1,1,1,1],stride=[1,1,1,1])
        self.conv1_3=convwithactivation4DRSTAR(cnum,cnum,kernel_size=[3,3,3,3],padding=[1,1,1,1],stride=[1,1,1,1])

        self.conv2=convwithactivation(cnum,2*cnum,kernel_size=[3,3,3],padding=[1,1,1],stride=[1,2,2],padding_mode='zeros')
        self.conv2_2=convwithactivation4DRSTAR(2*cnum,2*cnum,kernel_size=[3,3,3,3],padding=[1,1,1,1],stride=[1,1,1,1])
        self.conv2_3=convwithactivation4DRSTAR(2*cnum,2*cnum,kernel_size=[3,3,3,3],padding=[1,1,1,1],stride=[1,1,1,1])
        
        self.conv3=convwithactivation(2*cnum,4*cnum,kernel_size=[3,3,3],padding=[1,1,1],stride=[1,2,2],padding_mode='zeros')
        self.conv3_2=convwithactivation4DRSTAR(4*cnum,4*cnum,kernel_size=[3,3,3,3],padding=[1,1,1,1],stride=[1,1,1,1])
        self.conv3_3=convwithactivation4DRSTAR(4*cnum,4*cnum,kernel_size=[3,3,3,3],padding=[1,1,1,1],stride=[1,1,1,1])
        
        self.conv4=convwithactivation(4*cnum,8*cnum,kernel_size=[3,3,3],padding=[1,1,1],stride=[1,2,2],padding_mode='zeros')
        self.conv4_2=convwithactivation4DRSTAR(8*cnum,8*cnum,kernel_size=[3,3,3,3],padding=[1,1,1,1],stride=[1,1,1,1])
        self.conv4_3=convwithactivation4DRSTAR(8*cnum,8*cnum,kernel_size=[3,3,3,3],padding=[1,1,1,1],stride=[1,1,1,1])
        
        self.conv5=upconvwithactivation(8*cnum,4*cnum,kernel_size=[3,3,3,3],stride=[1,1,1,1],padding=[1,1,1,1],scale_factor=[1,1,2,2])
        self.conv5_2=convwithactivation4DRSTAR(8*cnum,4*cnum,kernel_size=[3,3,3,3],padding=[1,1,1,1],stride=[1,1,1,1])
        self.conv5_3=convwithactivation4DRSTAR(4*cnum,4*cnum,kernel_size=[3,3,3,3],padding=[1,1,1,1],stride=[1,1,1,1])
        
        self.conv6=upconvwithactivation(4*cnum,2*cnum,kernel_size=[3,3,3,3],stride=[1,1,1,1],padding=[1,1,1,1],scale_factor=[1,1,2,2])
        self.conv6_2=convwithactivation4DRSTAR(4*cnum,2*cnum,kernel_size=[3,3,3,3],padding=[1,1,1,1],stride=[1,1,1,1])
        self.conv6_3=convwithactivation4DRSTAR(2*cnum,2*cnum,kernel_size=[3,3,3,3],padding=[1,1,1,1],stride=[1,1,1,1])
        
        self.conv7=upconvwithactivation(2*cnum,1*cnum,kernel_size=[3,3,3,3],stride=[1,1,1,1],padding=[1,1,1,1],scale_factor=[1,1,2,2])
        self.conv7_2=convwithactivation4DRSTAR(2*cnum,1*cnum,kernel_size=[3,3,3,3],padding=[1,1,1,1],stride=[1,1,1,1])
        self.conv7_3=convwithactivation4DRSTAR(1*cnum,1*cnum,kernel_size=[3,3,3,3],padding=[1,1,1,1],stride=[1,1,1,1])
        
        ##output
        self.conv8=upconvwithactivation(1*cnum,1,kernel_size=[3,3,3,3],stride=[1,1,1,1],padding=[1,1,1,1],scale_factor=[1,2,1,1],ifrelu=0)

        self.motion = None
        self.prior_fusion = None
        self.activation_checkpointing = False

    def _backbone_step(self, function, *args):
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(function,*args,use_reentrant=False)
        return function(*args)

    def forward(self,x,*,context=None,return_aux=False):
        image = x
        x=self._backbone_step(_encode,x,self.conv1,self.conv1_2,self.conv1_3)
        x1=self._backbone_step(_encode,x,self.conv2,self.conv2_2,self.conv2_3)
        x2=self._backbone_step(_encode,x1,self.conv3,self.conv3_2,self.conv3_3)
        aux = {}
        enhanced = x2
        if self.motion is not None:
            delta, aux = self.motion(image[:,:1],x2,context,return_aux)
            enhanced = enhanced + delta
        if self.prior_fusion is not None:
            enhanced = enhanced + self.prior_fusion(image[:,:1],image[:,1:2,0],context)
        x2 = enhanced
        x3=self._backbone_step(_encode,x2,self.conv4,self.conv4_2,self.conv4_3)
        x3=self._backbone_step(_decode,x3,x2,self.conv5,self.conv5_2,self.conv5_3)
        x3=self._backbone_step(_decode,x3,x1,self.conv6,self.conv6_2,self.conv6_3)
        x3=self._backbone_step(_decode,x3,x,self.conv7,self.conv7_2,self.conv7_3)
        x3=self._backbone_step(self.conv8,x3)

        return (x3,aux) if return_aux else x3
