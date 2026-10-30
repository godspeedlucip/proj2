import torch
from torch.utils.data import DataLoader
import numpy as np
import tifffile
import random

# class npz_reader_2d_sim_label_real(torch.utils.data.Dataset):
#     def __init__(self, tripled_data_txt):
#         super(npz_reader_2d_sim_label_real, self).__init__()
#         self.tripled_files = open(tripled_data_txt).readlines()

#     def __len__(self):
#         return len(self.tripled_files)

#     def to_tensor(self, data):
#         if data.ndim == 2:
#             data = data[np.newaxis, ...]
#         elif data.ndim == 3:
#             pass
#         data = torch.FloatTensor(data)

#         return data

#     def to_numpy(self, data):
#         data = data.detach().cpu().numpy()
#         data = data.squeeze()
#         if data.ndim == 3:
#             data = data.transpose(1, 2, 0)

#         return data

#     def get(self, tripled_files):
#         npzData = np.load(tripled_files[:-1])

#         inputData = npzData['sim']
#         labelData = npzData['label']
#         realCT = npzData['real']

#         inputData = np.array(inputData).astype(np.float32) * 50
#         labelData = np.array(labelData).astype(np.float32) * 50
#         realCT = np.array(realCT).astype(np.float32) * 50
#         inputData[inputData < 0] = 0
#         labelData[labelData < 0] = 0
#         realCT[realCT < 0] = 0
#         inputData = self.to_tensor(inputData)
#         labelData = self.to_tensor(labelData)
#         realCT = self.to_tensor(realCT)

#         return {'sim': inputData, 'label': labelData, 'real': realCT}

#     def __getitem__(self, idx):
#         tripled_files = self.tripled_files[idx]
#         return self.get(tripled_files)

# class npz_reader_2d_sim_label_real_ref(torch.utils.data.Dataset):
#     def __init__(self, quadruple_data_txt):
#         super(npz_reader_2d_sim_label_real_ref, self).__init__()
#         self.quadruple_files = open(quadruple_data_txt).readlines()

#     def __len__(self):
#         return len(self.quadruple_files)

#     def to_tensor(self, data):
#         if data.ndim == 2:
#             data = data[np.newaxis, ...]
#         elif data.ndim == 3:
#             pass
#         data = torch.FloatTensor(data)

#         return data

#     def to_numpy(self, data):
#         data = data.detach().cpu().numpy()
#         data = data.squeeze()
#         if data.ndim == 3:
#             data = data.transpose(1, 2, 0)

#         return data

#     def get(self, quadruple_files):
#         npzData = np.load(quadruple_files[:-1])

#         inputData = npzData['sim']
#         labelData = npzData['label']
#         realCT = npzData['real']
#         refCT = npzData['ref']

#         inputData = np.array(inputData).astype(np.float32) * 50
#         labelData = np.array(labelData).astype(np.float32) * 50
#         realCT = np.array(realCT).astype(np.float32) * 50
#         refCT = np.array(refCT).astype(np.float32) * 50
#         inputData[inputData < 0] = 0
#         labelData[labelData < 0] = 0
#         realCT[realCT < 0] = 0
#         refCT[refCT < 0] = 0
#         inputData = self.to_tensor(inputData)
#         labelData = self.to_tensor(labelData)
#         realCT = self.to_tensor(realCT)
#         refCT = self.to_tensor(refCT)

#         return {'sim': inputData, 'label': labelData, 'real': realCT, 'ref':refCT}

#     def __getitem__(self, idx):
#         quadruple_files = self.quadruple_files[idx]
#         return self.get(quadruple_files)

class npz_reader_2d_data_with_prior(torch.utils.data.Dataset):
    def __init__(self, data_with_prior_txt):
        super(npz_reader_2d_data_with_prior, self).__init__()
        self.denoised_data_with_prior_files = open(data_with_prior_txt).readlines()

    def __len__(self):
        return len(self.denoised_data_with_prior_files)

    def to_tensor(self, data):
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        elif data.ndim == 3:
            pass
        data = torch.FloatTensor(data)

        return data

    def to_numpy(self, data):
        data = data.detach().cpu().numpy()
        data = data.squeeze()
        if data.ndim == 3:
            data = data.transpose(1, 2, 0)

        return data

    def get(self, denoised_data_with_prior_files):
        npzData = np.load(denoised_data_with_prior_files[:-1])

        inputData = npzData['input']
        labelData = npzData['label']
        priorData = npzData['prior']

        scale_factor = np.random.uniform(0.9, 1.1)
        inputData = np.array(inputData).astype(np.float32) * 50 * scale_factor
        labelData = np.array(labelData).astype(np.float32) * 50 * scale_factor
        priorData = np.array(priorData).astype(np.float32) * 50 * scale_factor

        # inputData = np.array(inputData).astype(np.float32) * 50
        # labelData = np.array(labelData).astype(np.float32) * 50
        # priorData = np.array(priorData).astype(np.float32) * 50

        inputData[inputData < 0] = 0
        labelData[labelData < 0] = 0
        priorData[priorData < 0] = 0

        inputData = self.to_tensor(inputData)
        labelData = self.to_tensor(labelData)
        priorData = self.to_tensor(priorData)

        return {'input': inputData, 'label': labelData, 'prior': priorData}

    def __getitem__(self, idx):
        denoised_data_with_prior = self.denoised_data_with_prior_files[idx]
        return self.get(denoised_data_with_prior)

class npz_reader_2d_adv_data_with_prior(torch.utils.data.Dataset):
    def __init__(self, sim_data_with_prior_txt, adv_data_with_prior_txt):
        super(npz_reader_2d_adv_data_with_prior, self).__init__()
        self.sim_data_with_prior_files = open(sim_data_with_prior_txt).readlines()
        self.adv_data_with_prior_files = open(adv_data_with_prior_txt).readlines()

    def __len__(self):
        return len(self.sim_data_with_prior_files)

    def to_tensor(self, data):
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        elif data.ndim == 3:
            pass
        data = torch.FloatTensor(data)

        return data

    def to_numpy(self, data):
        data = data.detach().cpu().numpy()
        data = data.squeeze()
        if data.ndim == 3:
            data = data.transpose(1, 2, 0)

        return data

    def get(self, sim_data_with_prior_files, adv_data_with_prior_files):
        npzData_sim = np.load(sim_data_with_prior_files[:-1])
        npzData_adv = np.load(adv_data_with_prior_files[:-1])

        inputData = npzData_sim['input']
        labelData = npzData_sim['label']
        simpriorData = npzData_sim['prior']
        realCT = npzData_adv['real_input']
        realpriorData = npzData_adv['real_prior']

        scale_factor = np.random.uniform(0.9, 1.1)
        inputData = np.array(inputData).astype(np.float32) * 50 * scale_factor
        labelData = np.array(labelData).astype(np.float32) * 50 * scale_factor
        simpriorData = np.array(simpriorData).astype(np.float32) * 50 * scale_factor
        realCT = np.array(realCT).astype(np.float32) * 50 * scale_factor
        realpriorData = np.array(realpriorData).astype(np.float32) * 50 * scale_factor

        inputData[inputData < 0] = 0
        labelData[labelData < 0] = 0
        simpriorData[simpriorData < 0] = 0
        realCT[realCT < 0] = 0
        realpriorData[realpriorData < 0] = 0
        # tifffile.imwrite(f'./npz_{scale_factor}.tif', np.concatenate([inputData, labelData, simpriorData, realCT, realpriorData], axis=0))

        inputData = self.to_tensor(inputData)
        labelData = self.to_tensor(labelData)
        simpriorData = self.to_tensor(simpriorData)
        realCT = self.to_tensor(realCT)
        realpriorData = self.to_tensor(realpriorData)

        return {'sim_input': inputData, 'sim_label': labelData, 'sim_prior': simpriorData,
                'real_input': realCT, 'real_prior': realpriorData}

    def __getitem__(self, idx):
        adv_choice = random.randint(0, len(self.adv_data_with_prior_files) - 1)
        sim_data_with_prior = self.sim_data_with_prior_files[idx]
        adv_data_with_prior = self.adv_data_with_prior_files[adv_choice]
        return self.get(sim_data_with_prior, adv_data_with_prior)