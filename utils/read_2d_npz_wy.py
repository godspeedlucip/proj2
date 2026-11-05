import torch
from torch.utils.data import DataLoader
import numpy as np

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
    def __init__(self, data_with_prior_txt: str, *, training: bool = True,
                 require_mask: bool = False, return_case_id: bool = False,
                 mask_labels=None):
        super(npz_reader_2d_data_with_prior, self).__init__()
        self.denoised_data_with_prior_files = []
        self.quality_overrides = []
        self.case_ids = []
        with open(data_with_prior_txt, encoding='utf-8') as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                fields = [field.strip() for field in line.rstrip('\r\n').split('\t')]
                if (len(fields) > 3 or not fields[0]
                        or (len(fields) >= 2 and fields[1] not in ('', '0', '1'))):
                    raise ValueError(f'{data_with_prior_txt}:{line_number}: expected path, optional quality 0/1, optional case_id')
                self.denoised_data_with_prior_files.append(fields[0])
                self.quality_overrides.append(int(fields[1]) if len(fields) >= 2 and fields[1] else None)
                self.case_ids.append(fields[2] if len(fields) == 3 and fields[2] else None)
        self.training = training
        self.require_mask = require_mask
        self.return_case_id = return_case_id
        if mask_labels is not None:
            if (not isinstance(mask_labels, (list, tuple)) or not mask_labels
                    or any(isinstance(value, bool) or not isinstance(value, (int, float))
                           or not np.isfinite(value) or value <= 0 for value in mask_labels)):
                raise ValueError('mask_labels must be a nonempty list of positive numbers')
        self.mask_labels = mask_labels

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

    def get(self, path, quality_override=None, case_id=None):
        with np.load(path, allow_pickle=False) as npzData:
            inputData = npzData['input']
            labelData = npzData['label']
            priorData = npzData['prior']
            if self.require_mask:
                if 'mask' not in npzData:
                    raise ValueError(f'{path}: missing mask')
                maskData = np.asarray(npzData['mask'])
                if maskData.ndim != 2 or maskData.shape != labelData.shape[-2:]:
                    raise ValueError(f'{path}: mask shape does not match label')
                if not np.issubdtype(maskData.dtype, np.number) or not np.isfinite(maskData).all():
                    raise ValueError(f'{path}: mask has nonfinite or nonnumeric values')
                if (maskData < 0).any():
                    raise ValueError(f'{path}: mask labels must be nonnegative')
                maskData = (maskData > 0 if self.mask_labels is None
                            else np.isin(maskData, self.mask_labels))
                if not maskData.any() or maskData.all():
                    raise ValueError(f'{path}: selected mask must contain foreground and background')
                quality = -1
                if 'mask_quality' in npzData:
                    qualityData = np.asarray(npzData['mask_quality'])
                    if (qualityData.ndim != 0
                            or not np.issubdtype(qualityData.dtype, np.integer)
                            or qualityData.item() not in (0, 1)):
                        raise ValueError(f'{path}: mask_quality must be scalar 0 or 1')
                    quality = int(qualityData.item())
                if quality_override is not None:
                    if quality != -1 and quality != quality_override:
                        raise ValueError(f'{path}: mask_quality conflicts with list value')
                    quality = quality_override

        scale_factor = np.random.uniform(0.9, 1.1) if self.training else 1.0
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

        result = {'input': inputData, 'label': labelData, 'prior': priorData}
        if self.require_mask:
            result['mask'] = self.to_tensor(maskData.astype(np.float32))
            result['mask_quality'] = torch.tensor(quality, dtype=torch.int64)
        if self.return_case_id:
            if case_id is None:
                import re
                from pathlib import Path
                match = re.fullmatch(r'(.+)_p\d+_slice\d+', Path(path).stem)
                case_id = match.group(1) if match else None
            if not case_id:
                raise ValueError(f'{path}: case_id is required for validation; add a third list column')
            result['case_id'] = case_id
        return result

    def __getitem__(self, idx):
        denoised_data_with_prior = self.denoised_data_with_prior_files[idx]
        return self.get(denoised_data_with_prior, self.quality_overrides[idx], self.case_ids[idx])

# class npz_reader_2d_adv_data_with_prior(torch.utils.data.Dataset):
#     def __init__(self, adv_data_with_prior_txt):
#         super(npz_reader_2d_adv_data_with_prior, self).__init__()
#         self.adv_data_with_prior_files = open(adv_data_with_prior_txt).readlines()

#     def __len__(self):
#         return len(self.adv_data_with_prior_files)

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

#     def get(self, adv_data_with_prior_files):
#         npzData = np.load(adv_data_with_prior_files[:-1])

#         inputData = npzData['sim']
#         labelData = npzData['label']
#         simpriorData = npzData['sprior']
#         realCT = npzData['real']
#         refCT = npzData['ref']
#         realpriorData = npzData['rprior']

#         inputData = np.array(inputData).astype(np.float32) * 50
#         labelData = np.array(labelData).astype(np.float32) * 50
#         simpriorData = np.array(simpriorData).astype(np.float32) * 50
#         realCT = np.array(realCT).astype(np.float32) * 50
#         refCT = np.array(refCT).astype(np.float32) * 50
#         realpriorData = np.array(realpriorData).astype(np.float32) * 50

#         inputData[inputData < 0] = 0
#         labelData[labelData < 0] = 0
#         simpriorData[simpriorData < 0] = 0
#         realCT[realCT < 0] = 0
#         refCT[refCT < 0] = 0
#         realpriorData[realpriorData < 0] = 0

#         inputData = self.to_tensor(inputData)
#         labelData = self.to_tensor(labelData)
#         simpriorData = self.to_tensor(simpriorData)
#         realCT = self.to_tensor(realCT)
#         refCT = self.to_tensor(refCT)
#         realpriorData = self.to_tensor(realpriorData)

#         return {'sim': inputData, 'label': labelData, 'simPrior': simpriorData,
#                 'real': realCT, 'ref': refCT, 'realPrior': realpriorData}

#     def __getitem__(self, idx):
#        adv_data_with_prior = self.adv_data_with_prior_files[idx]
#        return self.get(adv_data_with_prior)
