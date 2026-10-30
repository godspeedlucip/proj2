import torch
from torch.utils.data import Dataset
import numpy as np

class npz_reader_2d(torch.utils.data.Dataset):
    def __init__(self, paired_data_txt):
        super(npz_reader_2d, self).__init__()
        self.paired_files = open(paired_data_txt).readlines()

    def __len__(self):
        return len(self.paired_files)

    def to_tensor(self, data):
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        elif data.ndim == 3:
            # data = data.transpose(2, 0, 1)
            # data = data[np.newaxis, ...]
            pass

        data = torch.FloatTensor(data)

        return data

    def to_numpy(self, data):
        data = data.detach().cpu().numpy()
        data = data.squeeze()
        if data.ndim == 3:
            data = data.transpose(1, 2, 0)

        return data
    
    def get(self, paired_files):

        npzData = np.load(paired_files[:-1])

        rdctData = npzData['label'] * 30
        ldctData = npzData['input'] * 30
        rdctData = np.array(rdctData).astype(np.float32)
        ldctData = np.array(ldctData).astype(np.float32)
        rdctData = self.to_tensor(rdctData)
        ldctData = self.to_tensor(ldctData)

        return {"rdct": rdctData, "ldct": ldctData}

    def __getitem__(self, index):
        paired_files = self.paired_files[index]
        return self.get(paired_files)
    
class npz_reader_2d_pi(torch.utils.data.Dataset):
    def __init__(self, paired_data_txt):
        super(npz_reader_2d_pi, self).__init__()
        self.paired_files = open(paired_data_txt).readlines()

    def __len__(self):
        return len(self.paired_files)

    def to_tensor(self, data):
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        elif data.ndim == 3:
            data = data.transpose(2, 0, 1)

        data = torch.FloatTensor(data)

        return data

    def to_numpy(self, data):
        data = data.detach().cpu().numpy()
        data = data.squeeze()
        if data.ndim == 3:
            data = data.transpose(1, 2, 0)

        return data
    
    def get(self, paired_files):

        npzData = np.load(paired_files[:-1])

        rdctData = npzData['label'] * 30
        ldctData = npzData['input'] * 30
        pictData = npzData['prior'] * 30
        rdctData = np.array(rdctData).astype(np.float32)
        ldctData = np.array(ldctData).astype(np.float32)
        pictData = np.array(pictData).astype(np.float32)
        rdctData = self.to_tensor(rdctData)
        ldctData = self.to_tensor(ldctData)
        pictData = self.to_tensor(pictData)

        return {"label": rdctData, 'prior': pictData, "input": ldctData}

    def __getitem__(self, index):
        paired_files = self.paired_files[index]
        return self.get(paired_files)
