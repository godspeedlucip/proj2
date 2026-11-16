%% RSTAR4D: Rotational Streak Artifact Reduction in 4-D CBCT Using Separable 4-D Convolutions
% Author: Ziheng Deng
% code for simulated 4D CBCT scan
% The code can be slightly different from the version we used for the original paper
% We use ASTRA toolbox to perform CT forward/back-projection
% Input: P_gt = 4D CT images, RespBin = Respiratory signals
% Output: P_recon_whole = 10 phases 4D CBCT images (uncorrected, the input of neural network), prior = averaged reconstruction
% Download sample data: https://drive.google.com/drive/folders/1--I0lfTO-eEdY7wMNMQdqbn-mWgCsTR1?usp=sharing
tic
file1=dir('sample_data\Raw4DCT\'); % we provide a data sample, which is a synthetic case
for i=[1]
    for aug=1
        RespBin = csvread('sample_data\Resp_signal\RespBin1.csv'); % Use different respiratory signals for different cases
        path=strcat('sample_data\Raw4DCT\',file1(i+2).name,'\',num2str(aug),'\');
        projection=zeros([512,384,680],'single');
        x_clip1=randi(10); % Random clip for data augmentation
        x_clip2=randi(10);
        y_clip1=randi(10);
        y_clip2=randi(10);
        y_translate=randi(180)-90; % To simulate real clinical scenarios with random shift
        rotateang=randi(30)-15;
        rotateaxis=[0.06-randi(120)/1000,0.06-randi(120)/1000,1];
        k=1.07-randi(14)/100;

        % Simulated 4D CBCT scan
        for phase=1:10
            aa=floor(phase/10);bb=mod(phase,10);
            load(strcat(file1(i+2).folder,'\',file1(i+2).name,'\phase',num2str(aa),num2str(bb))); % Simulate 4D CBCT scan from 4D CT images with 10 phases
            Z_size=320;
            P_gt=imtranslate(P_gt,[y_translate,0,0],'OutputView','same');
            P_gt=imresize3(P_gt,[1024,1024,Z_size]);
            P_gt=imrotate3(P_gt,rotateang,rotateaxis,'cubic','crop');
            P_gt=imresize3(P_gt(x_clip1*2+1:end-x_clip2*2,y_clip1*2+1:end-y_clip2*2,:),[512,512,Z_size]);
            P_gt=k*P_gt;
            projind=(RespBin==phase);projind=find(projind==1);
            
            P_size=512;  
            pixsize=512/512; 
            SOD=1000/pixsize;  % CT geometry, should be set according to your task
            DOD=500/pixsize;    
            samp_rate=680; % 680views/rotation

            u_width=1024*0.776/pixsize; % size of the detector
            u_num=1024;
            v_width=384*0.776/pixsize;
            v_num=384;
            proj_geom = astra_create_proj_geom('cone', u_width/u_num, v_width/v_num, v_num, u_num, -1*(projind-1)/samp_rate*2*pi, SOD, DOD);
            vol_geom = astra_create_vol_geom(P_size,P_size,Z_size);
            [proj_id, proj_data] = astra_create_sino3d_cuda(double(P_gt), proj_geom, vol_geom);
            astra_mex_data3d('delete', proj_id);

            projection(:,:,projind)=imresize3(permute(proj_data,[1,3,2]),[512,384,length(projind)]); % downsample for efficiency
            clear proj_data
            P_gt=imresize3(P_gt(49:end-48,49:end-48,71:250),[416,416,180]); % Crop to avoid cone-beam artifact
        end

        % add random noises
        px=(1+randi(60)/10)*10^4;
        atten=px*exp(-1*projection/1000/10*0.1707);
        for kk=1:size(atten,3)
            atten(:,:,kk)=single(10^6*imnoise(atten(:,:,kk)/10^6,'poisson'));
        end

        projection=-1*log(atten/px);
        projection=projection*1000*10/0.1707;

        % Ready for image reconstruction

        P_size=416;  
        Z_size=180;
        u_num=512;

        prior=zeros([P_size,P_size,Z_size],'single');
        P_recon_whole=zeros([P_size,P_size,Z_size,10],'single');
        for phase=1:10
            aa=floor(phase/10);bb=mod(phase,10);
            projind=(RespBin==phase);projind=find(projind==1);

            vol_geom = astra_create_vol_geom(P_size,P_size,Z_size);
            proj_geom = astra_create_proj_geom('cone', u_width/u_num, v_width/v_num, v_num, u_num, -1*(projind-1)/samp_rate*2*pi, SOD, DOD);
            proj_id = astra_mex_data3d('create', '-proj3d', proj_geom, permute(projection(:,:,projind),[1,3,2]));
            rec_id = astra_mex_data3d('create', '-vol', vol_geom);
            cfg1 = astra_struct('FDK_CUDA');
            cfg1.ReconstructionDataId = rec_id;
            cfg1.ProjectionDataId = proj_id;
            alg1_id = astra_mex_algorithm('create', cfg1);
            astra_mex_algorithm('run', alg1_id);
            P_recon=astra_mex_data3d('get', rec_id);
            P_recon(isnan(P_recon))=0;
            prior=prior+P_recon*length(projind)/samp_rate;
            astra_mex_algorithm('delete', alg1_id);
            astra_mex_data3d('delete', rec_id);
            astra_mex_data3d('delete', proj_id);
            P_recon_whole(:,:,:,phase)=P_recon;
        end

        % visualization
        subplot(3,3,1);imshow(squeeze(P_gt(:,:,round(size(P_gt,3)/2))),[0,1500]);
        subplot(3,3,4);imshow(squeeze(P_gt(:,round(size(P_gt,2)/2),:))',[0,1500]);
        subplot(3,3,7);imshow(squeeze(P_gt(round(size(P_gt,1)/2),:,:))',[0,1500]);
        subplot(3,3,2);imshow(squeeze(P_recon(:,:,round(size(P_gt,3)/2))),[0,1500]);
        subplot(3,3,5);imshow(squeeze(P_recon(:,round(size(P_gt,2)/2),:))',[0,1500]);
        subplot(3,3,8);imshow(squeeze(P_recon(round(size(P_gt,1)/2),:,:))',[0,1500]);
        subplot(3,3,3);imshow(squeeze(prior(:,:,round(size(P_gt,3)/2))),[0,1500]);
        subplot(3,3,6);imshow(squeeze(prior(:,round(size(P_gt,2)/2),:))',[0,1500]);
        subplot(3,3,9);imshow(squeeze(prior(round(size(P_gt,1)/2),:,:))',[0,1500]);
        set(gcf,'position',[500,30,1000,1000])
        pause(.1)
    toc
    end
end



