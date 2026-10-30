import os
import tifffile
from utils.pytools import *
import numpy as np
import random
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

##################制作数据集方法##################### 

def make_2d_data_with_prior_list_txt(txt_file, sample_list, ldct_dir, rdct_dir, prior_dir, save_dir, train=True):
    make_dirs(save_dir)

    for sample_name in sample_list:

        if train:

            for phase in range(10):

                # print(f'{ldct_dir}/{sample_name}/{sample_name[:-2]}_{sample_name[-2:]}_p{phase}.tif')
                ldct_vol = tifffile.imread(f'{ldct_dir}/{sample_name}/{sample_name[:-2]}_{sample_name[-2:]}_p{phase}.tif')
                rdct_vol = tifffile.imread(f'{rdct_dir}/{sample_name}/{sample_name[:-2]}_{sample_name[-2:]}_p{phase}.tif')
                prior_vol = tifffile.imread(f'{prior_dir}/{sample_name}/{sample_name[:-2]}_{sample_name[-2:]}.tif')

                for index_z in range(np.size(rdct_vol, 0)): 

                    ldct_slice = ldct_vol[index_z, :, :]
                    rdct_slice = rdct_vol[index_z, :, :]
                    prior_slice = prior_vol[index_z, :, :]

                    # print(f'{save_dir}/{sample_name}_p{phase}_slice{index_z+1}')
                    np.savez(f'{save_dir}/{sample_name}_p{phase}_slice{index_z+1}', input=ldct_slice, label=rdct_slice, prior=prior_slice)
                    with open(txt_file, 'a') as f:
                        f.write(f'{save_dir}/{sample_name}_p{phase}_slice{index_z+1}.npz\n')
                    f.close()

        else:

            phase = random.randint(0, 9)
            # print(f'{ldct_dir}/{sample_name}/{sample_name[:-2]}_{sample_name[-2:]}_p{phase}.tif')
            ldct_vol = tifffile.imread(f'{ldct_dir}/{sample_name}/{sample_name[:-2]}_{sample_name[-2:]}_p{phase}.tif')
            rdct_vol = tifffile.imread(f'{rdct_dir}/{sample_name}/{sample_name[:-2]}_{sample_name[-2:]}_p{phase}.tif')
            prior_vol = tifffile.imread(f'{prior_dir}/{sample_name}/{sample_name[:-2]}_{sample_name[-2:]}.tif')

            for index_z in range(0, np.size(rdct_vol, 0), 4): 

                ldct_slice = ldct_vol[index_z, :, :]
                rdct_slice = rdct_vol[index_z, :, :]
                prior_slice = prior_vol[index_z, :, :]

                # print(f'{save_dir}/{sample_name}_p{phase}_slice{index_z+1}')
                np.savez(f'{save_dir}/{sample_name}_p{phase}_slice{index_z+1}', input=ldct_slice, label=rdct_slice, prior=prior_slice)
                with open(txt_file, 'a') as f:
                    f.write(f'{save_dir}/{sample_name}_p{phase}_slice{index_z+1}.npz\n')
                f.close()


def make_uda_data_with_prior_list_txt(txt_file, real_list, ldct_real_dir, prior_real_dir, save_dir):
    make_dirs(save_dir)
    for real_sample_name in real_list:
        for phase in range(10):
            # print(f'{ldct_dir}/{sample_name}/{sample_name[:-2]}_{sample_name[-2:]}_p{phase}.tif')
            # ldct_sim_vol = tifffile.imread(f'{ldct_sim_dir}/{sample_name}/{sample_name[:-2]}_{sample_name[-2:]}_p{phase}.tif')
            # rdct_sim_vol = tifffile.imread(f'{rdct_sim_dir}/{sample_name}/{sample_name[:-2]}_{sample_name[-2:]}_p{phase}.tif')
            # prior_sim_vol = tifffile.imread(f'{prior_sim_dir}/{sample_name}/{sample_name[:-2]}_{sample_name[-2:]}.tif')
            ldct_real_vol = tifffile.imread(f'{ldct_real_dir}/{real_sample_name}/{real_sample_name}_p{phase}.tif')
            prior_real_vol = tifffile.imread(f'{prior_real_dir}/{real_sample_name}/{real_sample_name}_prior.tif')
            for index_z in range(np.size(ldct_real_vol, 0)):
                # ldct_sim_slice = ldct_sim_vol[index_z, :, :]
                # rdct_sim_slice = rdct_sim_vol[index_z, :, :]
                # prior_sim_slice = prior_sim_vol[index_z, :, :]
                ldct_real_slice = ldct_real_vol[index_z, :, :]
                prior_real_slice = prior_real_vol[index_z, :, :]
                # print(f'{save_dir}/{sample_name}_p{phase}_slice{index_z+1}')
                # np.savez(f'{save_dir}/{sample_name}_p{phase}_slice{index_z+1}', input=ldct_sim_slice, label=rdct_sim_slice, prior=prior_sim_slice)
                np.savez(f'{save_dir}/{real_sample_name}_p{phase}_slice{index_z + 1}', real_input=ldct_real_slice, real_prior=prior_real_slice)
                tifffile.imwrite('./npz.tif', np.concatenate([ldct_real_slice, prior_real_slice], axis=0))
                with open(txt_file, 'a') as f:
                    f.write(f'{save_dir}/{real_sample_name}_p{phase}_slice{index_z+1}.npz\n')
                f.close()

##################制作#####################
if __name__ == '__main__':

    ###################### 文件路径 ##########################


    # 对抗训练集保存路径
    adv_train_save_dir = '/mnt/data/yikun/FDCBCT/TrainMC_v2/AdvTrain_v2'

    # 保存文件txt
    txt_save_dir = './txt/'
    make_dirs(txt_save_dir)
    # 对抗训练真实数据文件
    adv_train_txt = txt_save_dir + 'adv_train_MC_v2.txt'

    adv_sample_list = os.listdir("/mnt/data/yikun/FDCBCT/MC_HF_Fan_test/sparse_vol/")
    adv_sample_list.sort()

####################################################################################################################################################


    make_uda_data_with_prior_list_txt(txt_file=adv_train_txt, real_list=adv_sample_list,
                                      ldct_real_dir='/mnt/data/yikun/FDCBCT/MC_HF_Fan_test/sparse_vol/',
                                      prior_real_dir='/mnt/data/yikun/FDCBCT/MC_HF_Fan_test/prior_vol/',
                                      save_dir=adv_train_save_dir)


    with open(adv_train_txt, 'r') as infile:
        lines = infile.readlines()
    random.shuffle(lines)
    with open(adv_train_txt, 'w') as outfile:
        outfile.writelines(lines)
