########################################################################################################################
# BFA_train.py
########################################################################################################################

import os
import time
import datetime

import torch
import numpy as np
from BFA_ResUnet import BFA_resunet
from train_and_eval import train_one_epoch, evaluate, create_lr_scheduler
from my_dataset import VOCSegmentation
from callbacks import LossHistory
import transforms as T
import warnings
warnings.filterwarnings("ignore")

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class SegmentationPreset:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        trans = [T.Resize(224, 224)]                                                                                    # 尺寸
        trans.append(T.ToTensor())
        trans.append(T.Normalize(mean=mean, std=std))
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)

def get_transform():
    return SegmentationPreset()

def create_model(num_classes, pretrain=True):
    model =BFA_resunet(num_classes=num_classes, pretrain_backbone=True)
    if pretrain:
        model_path = '/content/drive/MyDrive/Project/save_weights_2/ep014-loss0.062-val_loss0.205.pth'                                                                # 预训练权重
        print('Load weights {}.'.format(model_path))
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    # segmentation nun_classes + background
    num_classes = args.num_classes + 1

    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    loss_history = LossHistory("logs/")

    train_dataset = VOCSegmentation(args.data_path,
                                    transforms=get_transform(), txt_name="train.txt")

    val_dataset = VOCSegmentation(args.data_path,
                                  transforms=get_transform(), txt_name="test.txt")

    num_workers = 2                                                                                                   

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=False,
                                               collate_fn=train_dataset.collate_fn,
                                               drop_last = True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=2,
                                             num_workers=num_workers,
                                             pin_memory=False,
                                             collate_fn=val_dataset.collate_fn,
                                             drop_last = True)

    model = create_model(num_classes=num_classes, pretrain=True)                                                       # 导入模型
    model.to(device)
    val_mean_loss, confmat = evaluate(model, val_loader, device=device, num_classes=num_classes)

    val_info = str(confmat)
    print(val_info)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch training")
    parser.add_argument("--data-path", default=r"data/", help="VOCdevkit root")                     # 数据集路径
    parser.add_argument("--num-classes", default=1, type=int)                                                           # 类别数，不包含背景
    parser.add_argument("--device", default="cuda", help="training device")                                             # 默认使用GPU
    parser.add_argument("-b", "--batch-size", default=16, type=int)                                                      # batch_size
    parser.add_argument("--epochs", default=50, type=int, metavar="N",                                                  # epochs
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')                               # 超参数；学习率
    parser.add_argument('--print-freq', default=100, type=int, help='print frequency')                                    # 打印频率
    parser.add_argument("--amp", default=True, type=bool,                                                               # 使用混合精度训练，较老显卡（如10系列）不支持，需要改为False
                        help="Use torch.cud"
                             "a.amp for mixed precision training")
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
