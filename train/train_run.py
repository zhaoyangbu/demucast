import argparse

import torch
from torch.utils.data import WeightedRandomSampler

from dataset.gtzan_dataset import GTZANDataset, GTZANDataset_baseline
from models.demucast_model import DemucastModel
from models.hdemucs_FE import HDemucs
from train.train_process import train
from torchsummary import summary


def get_args_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # train dataloader
    parser.add_argument('--label_dim', default=10, type=bool, metavar='LD', help='number of differnet classes')
    parser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('-w', '--num-workers', default=4
                        , type=int, metavar='NW',
                        help='# of workers for dataloading (default: 32)')


    # train args
    parser.add_argument('--baseline', default=False, type=bool, metavar='BL', help='use baseline trainning')
    parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument("--dataset", type=str, default="GTZAN", help="the dataset used")
    parser.add_argument("--num_epochs", type=int, default=80, help="number of maximum training epochs")
    parser.add_argument("--num_class", type=int, default=10, help="number of classes")
    parser.add_argument('--warmup_steps', type=int, default=800, metavar='N',
                        help='epochs to warmup LR')

    return parser

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    # base_train_loader = torch.utils.data.DataLoader(
    #     GTZANDataset_baseline('/data/z_projects/AWFE/data/datafiles/gtzan_tensor_train.json'),
    #     batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    #
    # base_val_loader = torch.utils.data.DataLoader(
    #     GTZANDataset_baseline('/data/z_projects/AWFE/data/datafiles/gtzan_tensor_val.json'),
    #     batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    train_loader = torch.utils.data.DataLoader(
        GTZANDataset('/data/z_projects/awfe_v2/data/datafiles/gtzan_tensor_train.json', pretrain=False, augmentation=False),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        GTZANDataset('/data/z_projects/awfe_v2/data/datafiles/gtzan_tensor_val.json', pretrain=False, augmentation=False),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)



    SOURCES = ["drums", "bass", "other", "vocals"]
    FE_model = HDemucs(sources=SOURCES, audio_channels=2)
    FE_dict = torch.load("/data/z_projects/awfe_v2/ckpt/demucs_model/e51eebcc-c1b80bdd.th")
    FE_model.load_state_dict(FE_dict['state'])

    model = DemucastModel(FE_model, args)
    print('Now starting training for {:d} epochs'.format(args.num_epochs))

    train(model, train_loader, val_loader, args)





print('training')