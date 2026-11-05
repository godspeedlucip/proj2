import os
from utils.read_2d_npz_wy import npz_reader_2d_data_with_prior
from torch.utils.data import DataLoader, SubsetRandomSampler
from utils.losses import *
from utils.utils import *
import random
import torch.nn
from torch.utils.data import DataLoader
from torch.backends import cudnn
import time
import torch.optim as optim
from utils.sophia import *
# from models import *
from datetime import datetime
from utils.pytools import *
import tifffile
from network_discriminator import *
# from priorNet import *
from PriorNet_pytorch import *
# from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train(train_loader, model, loss_mse, optimizer, scheduler, writer, epoch):
    batch_time = AverageMeter()
    loss_mse_scalar = AverageMeter()
    model.train()
    end = time.time()

    step = 0
    for data in tqdm(train_loader):
        
        inputImg = data['input']
        inputImg = inputImg.cuda()
        labelImg = data['label']
        labelImg = labelImg.cuda()
        priorImg = data['prior']
        priorImg = priorImg.cuda()

        # if epoch > 19:
        #     labelImg, inputImg = MixUp_AUG().aug(rgb_gt=labelImg, rgb_noisy=inputImg)
        _, predImg = model(inputImg, priorImg)

        fdkImgSave = predImg.data.cpu().numpy()
        fdkImgSave.astype(np.float32).tofile('predict.raw')

        loss1 = loss_mse(predImg, labelImg)
        loss = loss1

        loss_mse_scalar.update(loss1.item())
        batch_time.update(time.time() - end)
        end = time.time()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1

    writer.add_scalars('loss/mse', {'train_mse_loss': loss_mse_scalar.avg}, epoch + 1)
    writer.add_image('train img/reference img', normalization(labelImg[0, :, :, :]), epoch + 1)
    writer.add_image('train img/predict img', normalization(predImg[0, :, :, :]), epoch + 1)
    writer.add_image('train img/ldct img', normalization(inputImg[0, :, :, :]), epoch + 1)
    writer.add_image('train img/residual img',normalization(torch.abs(labelImg[0, :, :, :] - predImg[0, :, :, :])), epoch + 1)
    scheduler.step()

    print('Denoised Train Epoch: {}\t train_mse_loss: {:.6f}\t'.format(epoch + 1, loss_mse_scalar.avg))

def valid(valid_loader, model, loss_mse, loss_ssim, writer, epoch):
    batch_time = AverageMeter()
    loss_mse_scalar = AverageMeter()
    loss_ssim_scalar = AverageMeter()
    model.eval()
    end = time.time()

    for data in tqdm(valid_loader):
        inputImg = data['input']
        inputImg = inputImg.cuda()
        labelImg = data['label']
        labelImg = labelImg.cuda()
        priorImg = data['prior']
        priorImg = priorImg.cuda()

        with torch.no_grad():
            _, predictImg = model(inputImg, priorImg)
            loss1 = loss_mse(predictImg, labelImg)
            loss2 = loss_ssim(predictImg, labelImg)

        loss_mse_scalar.update(loss1.item())
        loss_ssim_scalar.update(loss2.item())
        batch_time.update(time.time() - end)
        end = time.time()

    writer.add_scalars('loss/mse', {'valid_mse_loss': loss_mse_scalar.avg}, epoch + 1)
    writer.add_scalars('loss/ssim', {'valid_ssim_loss': loss_ssim_scalar.avg}, epoch + 1)
    writer.add_image('valid img/reference img', normalization(labelImg[0, :, :, :]), epoch + 1)
    writer.add_image('valid img/predict img', normalization(predictImg[0, :, :, :]), epoch + 1)
    writer.add_image('valid img/ldct img', normalization(inputImg[0, :, :, :]), epoch + 1)
    writer.add_image('valid img/residual img', normalization(torch.abs(labelImg[0, :, :, :] - predictImg[0, :, :, :])), epoch + 1)

    print('Valid Epoch: {}\t valid_mse_loss: {:.6f}\t'.format(epoch + 1, loss_mse_scalar.avg))

def test_real(model, writer, epoch):

    model.eval()

    sample_list = os.listdir('/mnt/data/yikun/FDCBCT/RealLinaTechData_wy/Sparse')
    sample_name = np.random.choice(sample_list)
    phase = random.randint(0, 9)
    print(f'/mnt/data/yikun/FDCBCT/RealLinaTechData_wy/Sparse/{sample_name}/{sample_name}_p{phase}.tif')
    VolLDCT = tifffile.imread(f'/mnt/data/yikun/FDCBCT/RealLinaTechData_wy/Sparse/{sample_name}/{sample_name}_p{phase}.tif')
    VolPrior = tifffile.imread(f'/mnt/data/yikun/FDCBCT/RealLinaTechData_wy/Prior/{sample_name}_prior.tif')

    VolPredict = np.zeros_like(VolLDCT)

    for index in range(np.size(VolLDCT, 0)):

        LDCTImg = VolLDCT[index, :, :] * 50
        LDCTImg = LDCTImg[np.newaxis, np.newaxis, ...]
        LDCTImg = torch.FloatTensor(LDCTImg).cuda()

        PriorImg = VolPrior[index, :, :] * 50
        PriorImg = PriorImg[np.newaxis, np.newaxis, ...]
        PriorImg = torch.FloatTensor(PriorImg).cuda()

        with torch.no_grad():

            _, predictImg = model(LDCTImg, PriorImg)
            predictImg = np.squeeze(predictImg.data.cpu().numpy())

        VolPredict[index, :, :] = predictImg

    VolPredict.astype(np.float32).tofile('predict_real.raw')

    z_index = np.random.randint(0, VolPredict.shape[0])
    LDCTImgTB = VolLDCT[z_index, :, :]
    LDCTImgTB = LDCTImgTB[np.newaxis, np.newaxis, ...]
    LDCTImgTB = torch.FloatTensor(LDCTImgTB).cuda()

    predictImgTB = VolPredict[z_index, :, :]
    predictImgTB = predictImgTB[np.newaxis, np.newaxis, ...]
    predictImgTB = torch.FloatTensor(predictImgTB).cuda()

    predictImgTBLR = VolPredict[:, 240, :]
    predictImgTBLR = predictImgTBLR[np.newaxis, np.newaxis, ...]
    predictImgTBLR = torch.FloatTensor(predictImgTBLR).cuda()

    predictImgTBUD = VolPredict[:, :, 240]
    predictImgTBUD = predictImgTBUD[np.newaxis, np.newaxis, ...]
    predictImgTBUD = torch.FloatTensor(predictImgTBUD).cuda()

    writer.add_image('test img/predict img', normalization(predictImgTB[0, :, :, :]), epoch + 1)
    writer.add_image('test img/ldct img', normalization(LDCTImgTB[0, :, :, :]), epoch + 1)
    writer.add_image('test img/predict lr img', normalization(predictImgTBLR[0, :, :, :]), epoch + 1)
    writer.add_image('test img/predict ud img ', normalization(predictImgTBUD[0, :, :, :]), epoch + 1)

if __name__ == '__main__':

    cudnn.benchmark = True

    method = 'prior_net_sim_resp_0413_SophiaG_mixup'
    result_path = f'./runs/{method}/logs/'
    save_dir = f'./runs/{method}/checkpoints/'

    # get train dataset
    train_dataset = npz_reader_2d_data_with_prior(data_with_prior_txt='./txt/sim_train_v2.txt')
    train_subset_length = len(train_dataset) // 10

    # get sim_valid dataset
    sim_valid_dataset = npz_reader_2d_data_with_prior(data_with_prior_txt='./txt/sim_valid_v2.txt')
    sim_valid_dataloader = DataLoader(sim_valid_dataset, batch_size=40, num_workers=40, shuffle=True)

    model = Dense2D()
    # d_model = Discriminator_VGG_128_SN_modified()

    loss_mse = torch.nn.MSELoss()
    loss_ssim = SSIM()

    optimizer_g = SophiaG(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=2, gamma=0.95)

    if os.path.exists(save_dir) is False:

        model = model.cuda()

    else:
        checkpoint_latest = torch.load(find_lastest_file(save_dir))
        model = load_model(model, checkpoint_latest).cuda()
        optimizer_g.load_state_dict(checkpoint_latest['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint_latest['lr_scheduler'])
        print('Latest checkpoint {0} loaded.'.format(find_lastest_file(save_dir)))

    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    log_dir = os.path.join(result_path, time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    print("*"*20 + "Start Train" + "*"*20)

    for epoch in range(0, 100):
        train_indices = torch.randperm(len(train_dataset)).tolist()
        train_subset_indices = train_indices[:train_subset_length]
        train_sampler = SubsetRandomSampler(train_subset_indices)
        train_dataloader = DataLoader(train_dataset, batch_size=40, num_workers=40, sampler=train_sampler)

        print("*" * 20 + "Epoch: " + str(epoch + 1).rjust(4, '0') + "*" * 20)

        train(train_dataloader, model, loss_mse, optimizer_g, scheduler, writer, epoch)
        valid(sim_valid_dataloader, model, loss_mse, loss_ssim, writer, epoch)
        test_real(model, writer, epoch)

        save_model(model, optimizer_g, epoch + 1, save_dir, scheduler)

        # if epoch % 1000 == 0:
            # save_model(model, optimizer, epoch + 1, save_dir, scheduler)