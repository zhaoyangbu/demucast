import torch
import torchaudio
import torchaudio.transforms as T
import pathlib
import json


def gen_path_list(path):
    p = pathlib.Path(path)
    path_list = []
    for lst in list(p.glob("**/*.wav")):
        path_list.append(str(lst))
    return path_list


def gen_pkl_list(path):
    p = pathlib.Path(path)
    path_list = []
    for lst in list(p.glob("*.pkl")):
        path_list.append(str(lst).split("/")[-1])
    #print(path_list)

    with open("/data/z_projects/awfe_v2/data/datafiles/gtzan_tensor.json", "w") as f:
        json.dump({'data': path_list}, f)
    print(f"finished processing GTZAN tensor data")


def gen_label(label):
    idx = label_list.index(label)
    return idx


def wave_padding(wave):
  len_padding = 480000 - len(wave[0])
  padding_tensor = torch.zeros(1,len_padding)
  # print(wave.shape)
  # print(padding_tensor.shape)
  wave = torch.cat((wave, padding_tensor), 1)
  return wave


def gen_tensor(path, resampler, transform):
    genre = path.split('/')[-2]
    wave, sr = torchaudio.load(path)
    wave = resampler(wave)
    if len(wave[0]) >= 480000:
        wave = wave[:,:480000]
    else:
        wave = wave_padding(wave)
    wave = wave.repeat(2,1)
    mel_specgram = torch.squeeze(transform(wave))
    mel_specgram = mel_specgram[:,:1024]
    mel_specgram = torch.transpose(mel_specgram, 0, 1)
    label = gen_label(genre)
    return wave, mel_specgram, label


def gen_tensor_pkl(list):
    for i in range(len(list)):
        temp_data = []
        name = list[i].split('/')[-1].split('.')[0] + list[i].split('/')[-1].split('.')[1]
        wave, mel_specgram, label = gen_tensor(list[i], resampler, transform)
        temp_data = [wave, mel_specgram, label]
        torch.save(temp_data, f"/data/z_projects/awfe_v2/data/pkl_data/{name}.pkl")
        print(f"finished processing {name} tensor data")


# def gen_tensor_aug_pkl(list):
#     for i in range(len(list)):
#         name = list[i].split('/')[-1].split('.')[0] + list[i].split('/')[-1].split('.')[1]
#         wave, mel_specgram, label = gen_tensor(list[i], resampler, transform)
#         for j in range(3):
#             temp_data = []
#             temp_wave = wave[:,j*160000:(j+1)*160000]
#             temp_spec = mel_specgram[j*1024:(j+1)*1024,:]
#             temp_label = label
#             temp_data = [temp_wave, temp_spec, temp_label]
#             torch.save(temp_data, f"/Users/zhaoyangbu/Projects/awfe_v2/data/pkl_aug_data/{name}_{j}.pkl")
#             print(f"finished processing {name}_{j} tensor data")


def gen_data_list():
    val_lst = []
    train_lst = []
    for label in label_list:
        for i in range(20):
            val_name = label+"{:05n}".format(i)+'.pkl'
            val_lst.append(val_name)
        with open("/data/z_projects/awfe_v2/data/datafiles/gtzan_tensor_val.json", "w") as f:
            json.dump({'data': val_lst}, f)
        print(f"finished generating GTZAN val tensor data")

        for j in range(80):
            idx = 20+j
            train_name = label + "{:05n}".format(idx) + '.pkl'
            train_lst.append(train_name)
        with open("/data/z_projects/awfe_v2/data/datafiles/gtzan_tensor_train.json", "w") as f:
            json.dump({'data': train_lst}, f)
        print(f"finished generating GTZAN train tensor data")


# def gen_data_aug_list():
#     val_lst = []
#     train_lst = []
#     for label in label_list:
#         for i in range(60):
#             val_name = label+"{:05n}".format(i)+'_1.pkl'
#             train_name_1 = label + "{:05n}".format(i) + '_0.pkl'
#             train_name_2 = label + "{:05n}".format(i) + '_2.pkl'
#             val_lst.append(val_name)
#             train_lst.append(train_name_1)
#             train_lst.append(train_name_2)
#         with open("/Users/zhaoyangbu/Projects/awfe_v2/data/datafiles/gtzan_tensor_val_aug.json", "w") as f:
#             json.dump({'data': val_lst}, f)
#         print(f"finished generating GTZAN val aug tensor data")
#
#         for j in range(40):
#             idx = 60+j
#             for k in range(3):
#                 train_name = label + "{:05n}".format(idx) + f"_{k}"+'.pkl'
#                 train_lst.append(train_name)
#         with open("/Users/zhaoyangbu/Projects/awfe_v2/data/datafiles/gtzan_tensor_train_aug.json", "w") as f:
#             json.dump({'data': train_lst}, f)
#         print(f"finished generating GTZAN train aug tensor data")



if __name__ == "__main__":
    raw_sr = 22050
    down_sr = 16000
    resampler = T.Resample(raw_sr, down_sr)
    transform = T.MelSpectrogram(sample_rate=down_sr, n_fft=938, normalized=True, f_min =20, f_max=20000)

    path_1 = pathlib.Path("/data/z_projects/GTZAN")
    path_2 = pathlib.Path("/data/z_projects/awfe_v2/data/pkl_data")
    label_list = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']


    path_list = gen_path_list(path_1)
    gen_tensor_pkl(path_list)
    # gen_tensor_aug_pkl(path_list)
    gen_pkl_list(path_2)
    gen_data_list()
    # gen_data_aug_list()





