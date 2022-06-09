import json
import random
import torch
from torch.utils.data import Dataset, DataLoader



class GTZANDataset(Dataset):
    def __init__(self, name_path: str, val=False, pretrain=True, augmentation=False):
        self.name_path = name_path
        with open(name_path, 'r') as fp:
            data_json = json.load(fp)
        self.data_name = data_json['data']
        random.shuffle(self.data_name)
        # print(self.data_name)
        self.pretrain = pretrain
        self.augmentation = augmentation


    def __getitem__(self, index):
        if self.augmentation == False:
            tensor_path = f"/data/z_projects/awfe_v2/data/pkl_data/{self.data_name[index]}"
        else:
            tensor_path = f"/data/z_projects/awfe_v2/data/pkl_aug_data/{self.data_name[index]}"
        tensors = torch.load(tensor_path)
        wave = tensors[0]
        if self.pretrain == True:
            label = tensors[1] # label = mel spectrogram
        else:
            label = tensors[2]
        return wave, label

    def __len__(self):
        return len(self.data_name)


class GTZANDataset_baseline(Dataset):
    def __init__(self, name_path: str, augmentation=False):
        self.name_path = name_path
        with open(name_path, 'r') as fp:
            data_json = json.load(fp)
        self.data_name = data_json['data']
        random.shuffle(self.data_name)
        # print(self.data_name)
        self.augmentation = augmentation


    def __getitem__(self, index):
        if self.augmentation == False:
            tensor_path = f"/data/z_projects/awfe_v2/data/pkl_data/{self.data_name[index]}"
        else:
            tensor_path = f"/data/z_projects/awfe_v2/data/pkl_aug_data/{self.data_name[index]}"
        tensors = torch.load(tensor_path)
        spec = tensors[1]
        label = tensors[2]

        return spec, label

    def __len__(self):
        return len(self.data_name)


if __name__ == '__main__':
    # train_loader = torch.utils.data.DataLoader(
    #     GTZANDataset_baseline('/data/z_projects/awfe_v2/data/datafiles/gtzan_tensor_val.json'),
    #     batch_size=800, shuffle=False, num_workers=1, pin_memory=True)

#
#     val_loader = torch.utils.data.DataLoader(
#         GTZANDataset_baseline('/data/z_projects/awfe_v2/data/datafiles/gtzan_tensor_val.json'),
#         batch_size=200, shuffle=False, num_workers=1, pin_memory=True)

    train_loader = torch.utils.data.DataLoader(
        GTZANDataset('/data/z_projects/awfe_v2/data/datafiles/gtzan_tensor_train.json', pretrain=False, augmentation=False),
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True)


    # val_loader = torch.utils.data.DataLoader(
    #     GTZANDataset('/data/z_projects/data/datafiles/gtzan_tensor_val.json', pretrain=False),
    #     batch_size=200, shuffle=False, num_workers=1, pin_memory=True)


    train_features, train_labels = next(iter(train_loader))
    # val_features, val_labels, val_path = next(iter(val_loader))
    print(f"Feature batch shape: {train_features.size()}") #torch.Size([100, 2, 480000])
    print(f"Labels batch shape: {train_labels.size()}")
    # print(torch.isnan(train_features).any())
    # print(torch.isnan(train_labels).any())
    # lst = []
    # [lst.append(i) for i in train_path if not i in lst]
    # [lst.append(i) for i in val_path if not i in lst]
    # print(len(lst))

    



    # for i, (audio_input, labels) in enumerate(gtzan_loader):
    #     pass

