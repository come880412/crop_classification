import argparse
from cProfile import label
from turtle import forward
from dataset import Crop_data
from torch.utils.data import DataLoader
import torch
import tqdm
import os
import numpy as np
import csv

from utils import get_acc, get_wp_f1
from model import *
import torchvision.models as models
import torch.nn as nn


import warnings
warnings.filterwarnings("ignore")

data_dict = {0:'banana', 1:'bareland', 2:'carrot', 3:'corn', 4:'dragonfruit', 5:'garlic',
             6:'guava', 7:'peanut', 8:'pineapple', 9:'pumpkin', 10:'rice',
             11:'soybean', 12:'sugarcane', 13:'tomato'}

class ensemble_net(nn.Module):
    def __init__(self, model_list):
        super().__init__()
        self.model_list = model_list
        self.softmax = nn.Softmax()
    def forward(self, x):
        for idx, model in enumerate(self.model_list):
            if idx == 0:
                pred = self.softmax(model(x)).unsqueeze(1)
                print(pred)
            else:
                pred = torch.cat((pred, self.softmax(model(x)).unsqueeze(1)), dim=1)
        return torch.mean(pred, dim = 1)


def valid(opt, model, criterion, val_loader):
    model.eval()

    y_true = torch.tensor([]).type(torch.int16)
    y_pred = torch.tensor([]).type(torch.int16)
    total_correct = 0
    total_label = 0
    val_loss = 0.
    pbar = tqdm.tqdm(total=len(val_loader), ncols=0, desc="%s" % opt.mode, unit=" step")

    # save_list = [['image_filename', 'label']]
    save_list = [['image_filename', 'pred_label', 'true_label']]
    for images, labels, data_names in val_loader:
        with torch.no_grad():
            images, labels = images.cuda(), labels.cuda()

            if opt.model == 'cnn_pvt_fusion':
                out_CNN, out_PVT, pred = model(images)
                loss_CNN = criterion(out_CNN, label)
                loss_PVT = criterion(out_PVT, label)
                loss_all = criterion(pred, label)
                loss = 0.5 * loss_CNN + 0.5 * loss_PVT + loss_all
            else:
                preds = model(images)
                if opt.model == 'ViT' or opt.model == 'beit':
                    pred = pred.logits
            loss = criterion(preds, labels)

            correct, total = get_acc(preds, labels)

            total_label += total
            total_correct += correct
            val_acc = (total_correct / total_label) * 100

            val_loss += loss

            labels = labels.cpu().detach()
            preds = preds.cpu().detach()
            y_true = torch.cat((y_true, labels), 0)
            y_pred = torch.cat((y_pred, preds), 0)

            pbar.update()
            pbar.set_postfix(
                loss=f"{val_loss:.4f}",
                Accuracy=f"{val_acc:.2f}"
            )

            preds = preds.numpy()
            labels = labels.numpy()

            pred_label = np.argmax(preds, axis=1)
            for i in range(len(data_names)):
                data_name = data_names[i]
                pred = pred_label[i]
                true_label = labels[i]
                if pred != true_label:
                    save_list.append([data_name, str(data_dict[pred]), str(data_dict[true_label])])
                # save_list.append([data_name, str(data_dict[pred])])
            
    f1_dict, WP_value = get_wp_f1(y_pred, y_true)

    pbar.set_postfix(
        loss=f"{val_loss:.4f}",
        Accuracy=f"{val_acc:.2f}",
        WP_value=f"{WP_value:.4f}"
    )
    pbar.close()
    print(f1_dict)
    
    np.savetxt(os.path.join(opt.save_path, '%s.csv' % opt.mode),  save_list, fmt='%s', delimiter=',')

def test(opt, model, test_loader):
    model.eval()
    pbar = tqdm.tqdm(total=len(test_loader), ncols=0, desc="%s" % opt.mode, unit=" step")

    save_list = [['image_filename', 'label']]
    for images, data_names in test_loader:
        with torch.no_grad():
            images = images.cuda()
            preds = model(images)

            preds = preds.cpu().detach()
            pbar.update()

            preds = preds.numpy()

            pred_label = np.argmax(preds, axis=1)
            for i in range(len(data_names)):
                data_name = data_names[i]
                pred = pred_label[i]
                save_list.append([data_name, str(data_dict[pred])])
            
    pbar.close()
    np.savetxt(os.path.join(opt.save_path, '%s.csv' % opt.mode),  save_list, fmt='%s', delimiter=',')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='../dataset/public', help='path to dataset')
    parser.add_argument('--num_classes', type=int, default=14, help='number of classes')
    parser.add_argument('--mode', type=str, default="test", help='valid/test')
    parser.add_argument('--save_path', type=str, default="./output_csv")
    parser.add_argument('--five_crop', type=bool, default=False, help='whether to use five_crop')

    parser.add_argument("--batch_size", type=int, default=2, help="batch_size")

    parser.add_argument('--model', default='ensemble', help='resnext50/convnext_base/convnext_small/ensemble')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu workers')
    parser.add_argument('--load', default='./checkpoints/resnet50/R50_model.pth', help='path to model to continue training')
    
    opt = parser.parse_args()
    os.makedirs(opt.save_path, exist_ok=True)

    test_data = Crop_data(opt, opt.root, opt.mode)
    test_loader = DataLoader(test_data, batch_size=opt.batch_size,shuffle=False, num_workers=opt.n_cpu)

    if opt.model == 'resnext50':
        model = models.resnext50_32x4d(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, opt.num_classes)
    elif opt.model == 'convnext_small':
        model = convnext_small(opt)
    elif opt.model == 'convnext_base':
        model = convnext_base(opt)
    elif opt.model == 'ensemble':
        print("Use ensemble model!!")
        model_list = []
        model = models.resnext50_32x4d(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, opt.num_classes)
        model.load_state_dict(torch.load('./checkpoints/resnext50/model_epoch15_acc99.58.pth'))
        model = model.cuda()
        model.eval()
        model_list.append(model)

        model = convnext_small(opt)
        model.load_state_dict(torch.load('./checkpoints/convnext_small/model_epoch12_acc99.78.pth'))
        model = model.cuda()
        model.eval()
        model_list.append(model)

        model = convnext_base(opt)
        model.load_state_dict(torch.load('./checkpoints/convnext_base/model_epoch11_acc99.70.pth'))
        model = model.cuda()
        model.eval()
        model_list.append(model)

        model = ensemble_net(model_list)
    
    if opt.load != '' and opt.model != 'ensemble':
        print(f'loading pretrained model from {opt.load}')
        model.load_state_dict(torch.load(opt.load))
    model = model.cuda()
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()

    if opt.mode == 'valid':
        valid(opt, model, criterion, test_loader)
    elif opt.mode == 'test':
        test(opt, model, test_loader)