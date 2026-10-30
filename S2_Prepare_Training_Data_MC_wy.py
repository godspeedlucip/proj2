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
        #
        if train:

            for phase in range(10):
                file_path = f'{ldct_dir}/{sample_name}/{sample_name[:-2]}_{sample_name[-2:]}_p{phase}.tif'
                if os.path.exists(file_path) and os.path.isfile(file_path):
                    print(f'{ldct_dir}/{sample_name}/{sample_name[:-2]}_{sample_name[-2:]}_p{phase}.tif')
                    ldct_vol = tifffile.imread(
                        f'{ldct_dir}/{sample_name}/{sample_name[:-2]}_{sample_name[-2:]}_p{phase}.tif')
                    rdct_vol = tifffile.imread(
                        f'{rdct_dir}/{sample_name}/{sample_name[:-2]}_{sample_name[-2:]}_p{phase}.tif')
                    prior_vol = tifffile.imread(f'{prior_dir}/{sample_name}/{sample_name[:-2]}_{sample_name[-2:]}.tif')

                    for index_z in range(np.size(rdct_vol, 0)):
                        ldct_slice = ldct_vol[index_z, :, :]
                        rdct_slice = rdct_vol[index_z, :, :]
                        prior_slice = prior_vol[index_z, :, :]

                        # print(f'{save_dir}/{sample_name}_p{phase}_slice{index_z+1}')
                        np.savez(f'{save_dir}/{sample_name}_p{phase}_slice{index_z + 1}', input=ldct_slice,
                                 label=rdct_slice, prior=prior_slice)
                        with open(txt_file, 'a') as f:
                            f.write(f'{save_dir}/{sample_name}_p{phase}_slice{index_z + 1}.npz\n')
                        f.close()
                else:
                    pass


        else:

            for phase in range(10):
                file_path = f'{ldct_dir}/{sample_name}/{sample_name[:-2]}_{sample_name[-2:]}_p{phase}.tif'
                if os.path.exists(file_path) and os.path.isfile(file_path):
                    print(f'{ldct_dir}/{sample_name}/{sample_name[:-2]}_{sample_name[-2:]}_p{phase}.tif')
                    ldct_vol = tifffile.imread(
                        f'{ldct_dir}/{sample_name}/{sample_name[:-2]}_{sample_name[-2:]}_p{phase}.tif')
                    rdct_vol = tifffile.imread(
                        f'{rdct_dir}/{sample_name}/{sample_name[:-2]}_{sample_name[-2:]}_p{phase}.tif')
                    prior_vol = tifffile.imread(f'{prior_dir}/{sample_name}/{sample_name[:-2]}_{sample_name[-2:]}.tif')

                    for index_z in range(0, np.size(rdct_vol, 0), 4):
                        ldct_slice = ldct_vol[index_z, :, :]
                        rdct_slice = rdct_vol[index_z, :, :]
                        prior_slice = prior_vol[index_z, :, :]

                        # print(f'{save_dir}/{sample_name}_p{phase}_slice{index_z+1}')
                        np.savez(f'{save_dir}/{sample_name}_p{phase}_slice{index_z + 1}', input=ldct_slice,
                                 label=rdct_slice, prior=prior_slice)
                        with open(txt_file, 'a') as f:
                            f.write(f'{save_dir}/{sample_name}_p{phase}_slice{index_z + 1}.npz\n')
                        f.close()
                else:
                    pass


# def make_sim_data_list_txt(txt_file, data_list, ldct_dir, rdct_dir, prior_dir, save_dir, type='sim'):
#     make_dirs(save_dir)
#     if type == 'sim' or type == 'sim_valid':
#         for data_name in data_list:
#             ldct_vol = tifffile.imread(ldct_dir + data_name)
#             rdct_vol = tifffile.imread(rdct_dir + data_name)
#             prior_vol = tifffile.imread(prior_dir + data_name[:22] + '.tif')

#             for index_z in range(np.size(ldct_vol, 0)):
#                 ldct_slice = ldct_vol[index_z, :, :]
#                 rdct_slice = rdct_vol[index_z, :, :]
#                 prior_slice = prior_vol[index_z, :, :]

#                 np.savez(save_dir + data_name[:-4] + '_slice' + str(index_z + 1), input=ldct_slice, label=rdct_slice,
#                          prior=prior_slice)
#                 with open(txt_file, 'a') as f:
#                     f.write(save_dir + data_name[:-4] + '_slice' + str(index_z + 1) + '.npz\n')
#                 f.close()
#     elif type == 'real_test':
#         for data_name in data_list:
#             for i in range(1, 11):
#                 ref_vol = tifffile.imread(rdct_dir + data_name)
#                 real_vol = tifffile.imread(ldct_dir + data_name[:-4] + '_P' + str(i) + '.tif')
#                 prior_vol = tifffile.imread(prior_dir + data_name)

#                 for index_z in range(np.size(ref_vol, 0)):
#                     ref_slice = ref_vol[index_z, :, :]
#                     real_slice = real_vol[index_z, :, :]
#                     prior_slice = prior_vol[index_z, :, :]

#                     np.savez(save_dir + data_name[:-4] + '_P' + str(i) + '_slice' + str(index_z + 1), input=real_slice,
#                              label=ref_slice, prior=prior_slice)
#                     with open(txt_file, 'a') as f:
#                         f.write(save_dir + data_name[:-4] + '_P' + str(i) + '_slice' + str(index_z + 1) + '.npz\n')
#                     f.close()


# def make_adv_data_list_txt(txt_file, data_list, save_dir, sim_dir=sp_sim_dir, label_dir=full_sim_dir,
#                          real_dir=sp_real_dir , ref_dir=full_real_dir):
#     make_dirs(save_dir)

#     real_list = os.listdir(full_real_dir)
#     real_list.sort()

#     for sim_name in data_list:
#         s = sim_name[7:22]
#         phase = sim_name[24:-4]
#         random_choice = np.random.randint(0, 5)

#         sim_vol = tifffile.imread(sim_dir + sim_name)
#         label_vol = tifffile.imread(label_dir + sim_name)
#         sim_prior_vol = tifffile.imread(prior_sim_dir + sim_name[:22] + '.tif')

#         real_vol = tifffile.imread(real_dir + real_list[random_choice][:-4] + f'_P{phase}.tif')
#         ref_vol = tifffile.imread(ref_dir + real_list[random_choice])
#         real_prior_vol = tifffile.imread(prior_real_dir + real_list[random_choice])

#         for index_z in range(np.size(sim_vol, 0)):
#             sim_slice = sim_vol[index_z, :, :]
#             label_slice = label_vol[index_z, :, :]
#             sim_prior_slice = sim_prior_vol[index_z, :, :]
#             real_slice = real_vol[index_z, :, :]
#             ref_slice = ref_vol[index_z, :, :]
#             real_prior_slice = real_prior_vol[index_z, :, :]

#             np.savez(save_dir + sim_name[:-4] + '_slice' + str(index_z + 1),
#                      sim=sim_slice, label=label_slice, sprior=sim_prior_slice,
#                      real=real_slice, ref=ref_slice, rprior=real_prior_slice)
#             with open(txt_file, 'a') as f:
#                 f.write(save_dir + sim_name[:-4] + '_slice' + str(index_z + 1) + '.npz\n')
#             f.close()

##################制作#####################
if __name__ == '__main__':

    ###################### 文件路径 ##########################

    # 仿真数据路径
    prior_sim_dir = '/mnt/data/yikun/FDCBCT/SimulatedMC_sim_resbin_v2/PriorVol/'
    full_sim_dir = '/mnt/data/yikun/FDCBCT/SimulatedMC_sim_resbin_v2/FullVol/'
    sp_sim_dir = '/mnt/data/yikun/FDCBCT/SimulatedMC_sim_resbin_v2/SparseVol/'

    # leitai data
    sp_real_dir = '/mnt/data/yikun/FDCBCT/MC_HF_Fan/MC_HF_Fan/sparse_vol/'
    # full_real_dir = '/mnt/no1/yuanjixin/withPrior/Data/Leitai/LeitaiVollAllPhase/'
    prior_real_dir = '/mnt/data/yikun/FDCBCT/MC_HF_Fan/MC_HF_Fan/prior_vol/'

    # 降噪训练集保存路径
    sim_train_save_dir = '/mnt/data/yikun/FDCBCT/TrainMC_v2/SimTrain'
    # 仿真验证集保存路径
    sim_valid_save_dir = '/mnt/data/yikun/FDCBCT/TrainMC_v2/SimValid'
    # 对抗训练集保存路径
    adv_train_save_dir = '/mnt/data/yikun/FDCBCT/TrainMC_v2/AdvTrain'
    # 真实测试集保存路径
    real_test_save_dir = '/mnt/data/yikun/FDCBCT/TrainMC_v2/RealTest'

    # 保存文件txt
    txt_save_dir = './txt/'
    make_dirs(txt_save_dir)
    # 降噪训练文件
    sim_train_txt = txt_save_dir + 'sim_train_MC_v2.txt'
    # 仿真验证文件
    sim_valid_txt = txt_save_dir + 'sim_valid_MC_v2.txt'
    # 对抗训练文件
    adv_train_txt = txt_save_dir + 'adv_train_MC_v2.txt'
    # 真实测试文件
    real_test_txt = txt_save_dir + 'real_test_MC_v2.txt'

    sim_sample_list = os.listdir(full_sim_dir)
    sim_sample_list.sort()

    # all_sample_list = os.listdir('/mnt/data/yikun/FDCBCT/NumericalPhantom')
    # all_sample_list.sort()
    # valid_sample_list = all_sample_list[2::13]
    # test_sample_list = all_sample_list[1::13]

    valid_sample_list = np.array([sim_sample_list[i:i+8] for i in range(0, len(sim_sample_list) - 8 + 1, 8*13)]).flatten()
    # test_sample_list = np.array([sim_sample_list[i+8*11:i+8*12] for i in range(0, len(sim_sample_list) - 8 + 1, 8*13)]).flatten()
    train_sample_list = [item for item in sim_sample_list if item not in valid_sample_list]
    print(len(valid_sample_list))
    # print(len(test_sample_list))
    print(len(train_sample_list))

    # print(train_sample_list[34*8:35*8])
    # aaa
    # train_sample_list = train_sample_list[34*8:35*8]
    
####################################################################################################################################################
    if os.path.exists(sim_train_txt) == False:
        make_2d_data_with_prior_list_txt(txt_file=sim_train_txt, sample_list=train_sample_list,
                                        ldct_dir='/mnt/data/yikun/FDCBCT/SimulatedMC_sim_resbin_v2/SparseVol',
                                        rdct_dir='/mnt/data/yikun/FDCBCT/SimulatedMC_sim_resbin_v2/FullVol',
                                        prior_dir='/mnt/data/yikun/FDCBCT/SimulatedMC_sim_resbin_v2/PriorVol',
                                        save_dir=sim_train_save_dir)


    if os.path.exists(sim_valid_txt) == False:
        make_2d_data_with_prior_list_txt(txt_file=sim_valid_txt, sample_list=valid_sample_list, 
                                         ldct_dir='/mnt/data/yikun/FDCBCT/SimulatedMC_sim_resbin_v2/SparseVol',
                                         rdct_dir='/mnt/data/yikun/FDCBCT/SimulatedMC_sim_resbin_v2/FullVol',
                                         prior_dir='/mnt/data/yikun/FDCBCT/SimulatedMC_sim_resbin_v2/PriorVol',
                                         save_dir=sim_valid_save_dir,
                                         train=False)

    # if os.path.exists(sim_train_txt) == False:
    #     make_2d_data_with_prior_list_txt(txt_file=sim_train_txt, sample_list=train_sample_list,
    #                                      ldct_dir='/mnt/data/yikun/FDCBCT/SimulatedLinaTechData/SparseVol',
    #                                      rdct_dir='/mnt/data/yikun/FDCBCT/SimulatedLinaTechData/FullVol',
    #                                      prior_dir='/mnt/data/yikun/FDCBCT/SimulatedLinaTechData/PriorVol',
    #                                      save_dir=sim_train_save_dir)




    with open(sim_train_txt, 'r') as infile:
        lines = infile.readlines()
    random.shuffle(lines)
    with open(sim_train_txt, 'w') as outfile:
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