import json
import librosa

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, dataloader


def get_label_index_dict(labels_indices_path):
    label_df = pd.read_csv(labels_indices_path)
    label_indices = {}
    for index, row in label_df.iterrows():
        i = row["index"]
        label = row["mid"]
        label_indices[label] = i
    return label_indices


def wave_padding(wave):
  len_padding = 220500 - len(wave)
  padding_tensor = torch.zeros(len_padding)
  # print(wave.shape)
  # print(padding_tensor.shape)
  wave = torch.cat((wave, padding_tensor))
  return wave


def audio_tensor_resampled(path):
    y, sr = librosa.load(path)
    tensor = torch.FloatTensor(y)
    if len(tensor) >= 220500:
        wave = tensor[:220500]
    else:
        wave = wave_padding(tensor)
    return wave


class AudiosetDataset(Dataset):
    def __init__(self, dataset_json_file: str, audio_conf: dict, label_csv: str):

        super(AudiosetDataset, self).__init__()
        self.audio_conf = audio_conf
        print('-' * 15 + 'the pretrain dataset'+ '-' * 15)

        # load the json
        self.data_path = dataset_json_file
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)
        self.data = data_json['data']
        self.audio_conf = audio_conf


        self.noise = self.audio_conf.get('noise')
        if self.noise:
            print('now use noise augmentation')


        print('Label Setting')
        self.label_num = self.audio_conf.get('label_num')
        self.index_dict = get_label_index_dict(label_csv)

    def _index2data(self, main_index):
        waveform = audio_tensor_resampled(self.data[main_index]['wav'])

        label_indices = np.zeros(self.label_num)
        for label_str in self.data[main_index]['labels'].split(','):
            label_indices[int(self.index_dict[label_str])] = 1.0

        label_indices = torch.FloatTensor(label_indices)


        return waveform, label_indices

    def __getitem__(self, index):
        waveform, label_indices = self._index2data(main_index=index)


        if self.noise == True:
            waveform = waveform + torch.rand(waveform.shape) * np.random.rand() / 10
            waveform = torch.roll(waveform, np.random.randint(-10, 10), 0)

        waveform = waveform.unsqueeze(1)
        waveform = waveform.transpose(-2, -1)

        label_indices = label_indices.unsqueeze(1)
        label_indices = label_indices.transpose(-2, -1)
        label_indices = label_indices.squeeze(-2)

        return waveform, label_indices

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    audio_conf = {'noise': True, 'label_num':527}

    val_loader = torch.utils.data.DataLoader(
        AudiosetDataset('/data/hhd_projects/ASMAE/data/datafiles/eval.json',
                        label_csv='/data/hhd_projects/ASMAE/data/segments_csv/class_labels_indices.csv',
                        audio_conf=audio_conf),
        batch_size=4, shuffle=False, num_workers=1, pin_memory=True)


    audio_input, labels = next(iter(val_loader))
    print(f"Feature batch shape: {audio_input.size()}")
    print(f"Labels batch shape: {labels.size()}")

