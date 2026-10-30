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
            prior_real_vol = tifffile.imread(f'{prior_real_dir}/{real_sample_name}_prior.tif')
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

    # 仿真数据路径
    sp_sim_dir = '/mnt/data/yikun/FDCBCT/SimulatedLinaTechData/PriorVol/'
    full_sim_dir = '/mnt/data/yikun/FDCBCT/SimulatedLinaTechData/FullVol/'
    prior_sim_dir = '/mnt/data/yikun/FDCBCT/SimulatedLinaTechData/SparseVol/'

    # leitai data
    sp_real_dir = '/mnt/data/yikun/FDCBCT/RealLinaTechData/Sparse/'
    # full_real_dir = '/mnt/no1/yuanjixin/withPrior/Data/Leitai/LeitaiVollAllPhase/'  # 仅full剔除2例
    prior_real_dir = '/mnt/data/yikun/FDCBCT/RealLinaTechData/Prior/'

    # 降噪训练集保存路径
    sim_train_save_dir = '/mnt/data/yikun/FDCBCT/TrainLinaTech/SimTrain'
    # 仿真验证集保存路径
    sim_valid_save_dir = '/mnt/data/yikun/FDCBCT/TrainLinaTech/SimValid'
    # 对抗训练集保存路径
    adv_train_save_dir = '/mnt/data/yikun/FDCBCT/TrainLinaTech/AdvTrain'
    # 对抗验证集保存路径
    adv_valid_save_dir = '/mnt/data/yikun/FDCBCT/TrainLinaTech/AdvValid'
    # 真实测试集保存路径
    real_test_save_dir = '/mnt/data/yikun/FDCBCT/TrainLinaTech/RealTest'

    # 保存文件txt
    txt_save_dir = './txt/'
    make_dirs(txt_save_dir)
    # 降噪训练文件
    sim_train_txt = txt_save_dir + 'sim_train.txt'
    # 仿真验证文件
    sim_valid_txt = txt_save_dir + 'sim_valid.txt'
    # 对抗训练真实数据文件
    adv_train_txt = txt_save_dir + 'adv_train_v2.txt'
    # 真实测试文件
    real_test_txt = txt_save_dir + 'real_test.txt'

    sim_sample_list = os.listdir(full_sim_dir)
    sim_sample_list.sort()

    # all_sample_list = os.listdir('/mnt/data/yikun/FDCBCT/NumericalPhantom')
    # all_sample_list.sort()
    # valid_sample_list = all_sample_list[2::13]
    # test_sample_list = all_sample_list[1::13]

    valid_sample_list = np.array([sim_sample_list[i:i+8] for i in range(0, len(sim_sample_list) - 8 + 1, 8*13)]).flatten()
    test_sample_list = np.array([sim_sample_list[i+8*11:i+8*12] for i in range(0, len(sim_sample_list) - 8 + 1, 8*13)]).flatten()
    train_sample_list = [item for item in sim_sample_list if item not in valid_sample_list and item not in test_sample_list]
    # print(len(valid_sample_list))
    # print(len(test_sample_list))
    # print(len(train_sample_list))

    # print(train_sample_list[34*8:35*8])
    # aaa
    train_sample_list = train_sample_list[34*8:35*8]

    real_data_list = ['20230403_102523', '20230327_120743', '20230313_110413', '20230306_103620',
                      '20230206_111339', '20230128_101446', '20230111_100453', '20230109_105743']
####################################################################################################################################################


    # if os.path.exists(sim_train_txt) == False:
    #     make_2d_data_with_prior_list_txt(txt_file=sim_train_txt, sample_list=train_sample_list, 
    #                                      ldct_dir='/mnt/data/yikun/FDCBCT/SimulatedLinaTechData/SparseVol',
    #                                      rdct_dir='/mnt/data/yikun/FDCBCT/SimulatedLinaTechData/FullVol',
    #                                      prior_dir='/mnt/data/yikun/FDCBCT/SimulatedLinaTechData/PriorVol',
    #                                      save_dir=sim_train_save_dir)

    make_uda_data_with_prior_list_txt(txt_file=adv_train_txt, real_list=real_data_list,
                                      ldct_real_dir='/mnt/data/yikun/FDCBCT/RealLinaTechData_wy/Sparse/',
                                      prior_real_dir='/mnt/data/yikun/FDCBCT/RealLinaTechData_wy/Prior/',
                                      save_dir=adv_train_save_dir)


    with open(adv_train_txt, 'r') as infile:
        lines = infile.readlines()
    random.shuffle(lines)
    with open(adv_train_txt, 'w') as outfile:
        outfile.writelines(lines)

####################################################################################################################################################

    # # 制作去噪训练集
    # if os.path.exists(sim_train_txt) == False:
    #     make_sim_data_list_txt(txt_file=sim_train_txt, data_list=train_sample_list,
    #                                 ldct_dir=sp_sim_dir, rdct_dir=full_sim_dir, prior_dir=prior_sim_dir,
    #                                 save_dir=sim_train_save_dir, type='sim')
    # with open(sim_train_txt, 'r') as infile:
    #     lines = infile.readlines()
    # random.shuffle(lines)
    # with open(sim_train_txt, 'w') as outfile:
    #     outfile.writelines(lines)

    # print('开始制作仿真验证集')
    # # 制作仿真验证集
    # sim_valid_list = sample_list[-500:]
    # sim_valid_list = [s for s in sim_valid_list if 'S6' in s]
    # if os.path.exists(sim_valid_txt) == False:
    #     make_sim_data_list_txt(txt_file=sim_valid_txt, data_list=sim_valid_list,
    #                                 ldct_dir=sp_sim_dir, rdct_dir=full_sim_dir, prior_dir=prior_sim_dir,
    #                                 save_dir=sim_valid_save_dir, type='sim_valid')
    # with open(sim_valid_txt, 'r') as infile:
    #     lines = infile.readlines()
    # with open(sim_valid_txt, 'w') as outfile:
    #     outfile.writelines(lines)

    # print('开始制作对抗训练集')
    # # 制作对抗训练集，剔除较暗的2例，选用前5例
    # if os.path.exists(adv_train_txt) == False:
    #     make_adv_data_list_txt(txt_file=adv_train_txt, data_list=sim_train_list, save_dir=adv_train_save_dir)
    # with open(adv_train_txt, 'r') as infile:
    #     lines = infile.readlines()
    # random.shuffle(lines)
    # with open(adv_train_txt, 'w') as outfile:
    #     outfile.writelines(lines)

    # print('开始制作真实测试集')
    # # 制作真实测试集，选用剔除2例后的后3例
    # real_test_list = os.listdir(full_real_dir)
    # real_test_list.sort()
    # real_test_list = real_test_list[-3:]
    # # print(real_test_list)
    # if os.path.exists(real_test_txt) == False:
    #     make_sim_data_list_txt(txt_file=real_test_txt, data_list=real_test_list,
    #                                 ldct_dir=sp_real_dir, rdct_dir=full_real_dir, prior_dir=prior_real_dir,
    #                                 save_dir=real_test_save_dir, type='real_test')
    # with open(real_test_txt, 'r') as infile:
    #     lines = infile.readlines()
    # with open(real_test_txt, 'w') as outfile:
    #     outfile.writelines(lines)