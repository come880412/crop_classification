import imp
from turtle import forward
import torchvision.models as models
import torch.nn as nn
import torch

import argparse

from model_component.Resnet import ResNet, Bottleneck
from model_component.PVT import *
from model_component.convnext import ConvNeXt

class convnext_small(nn.Module):
    def __init__(self, opt):
        super(convnext_small, self).__init__()

        self.model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768])
        url = "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_1k_384.pth"
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"])

        self.model.head = nn.Linear(self.model.head.in_features, opt.num_classes)
    def forward(self, x):
        out, emb = self.model(x)
        return out

class convnext_base(nn.Module):
    def __init__(self, opt):
        super(convnext_base, self).__init__()

        self.model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
        url = "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_384.pth"
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"])

        self.model.head = nn.Linear(self.model.head.in_features, opt.num_classes)
    def forward(self, x):
        out, emb = self.model(x)
        return out

class convnext_large(nn.Module):
    def __init__(self, opt):
        super(convnext_large, self).__init__()

        self.model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536])
        url = "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_1k_384.pth"
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"])

        self.model.head = nn.Linear(self.model.head.in_features, opt.num_classes)
    def forward(self, x):
        out, emb = self.model(x)
        return out

class resnet18_emb(nn.Module):
    def __init__(self, opt):
        super(resnet18_emb, self).__init__()
        model = ResNet(block = Bottleneck, layers = [2, 2, 2, 2], num_classes = opt.num_classes)
        dict_ = models.resnet18(pretrained = True)
        dict_ = dict_.state_dict()
        model_dict = model.state_dict()
        pretrained_dict = {}
        for k, v in dict_.items():
            print(k)
            if k in model_dict and k.split('.')[0] != 'classifier':
                pretrained_dict.update({k: v})
            else:
                break
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        self.model = model
    def forward(self, x):
        out, emb = self.model(x)
        return out, emb

class resnet50(nn.Module):
    def __init__(self, opt):
        super(resnet50, self).__init__()
        model = ResNet(block = Bottleneck, layers = [3, 4, 6, 3], num_classes = opt.num_classes)
        dict_ = models.resnet50(pretrained = True)
        dict_ = dict_.state_dict()
        model_dict = model.state_dict()
        pretrained_dict = {}
        for k, v in dict_.items():
            if k in model_dict and k.split('.')[0] != 'fc':
                pretrained_dict.update({k: v})
            else:
                break
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        self.model = model
    def forward(self, x):
        out, emb = self.model(x)
        return out, emb

class attention(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.weight = nn.Sequential(
            nn.Linear(in_features= dim , out_features= dim // 4),
            nn.BatchNorm1d(dim // 4),
            nn.ReLU(),
            nn.Linear(in_features= dim // 4 , out_features= 1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
            input :
                x : (B, 6, dim)
            output :
                out : (B, dim)
        """
        _, seq_len, _ = x.size()
        image_ori = x[:,0,:]

        out = 0
        for i in range(1, seq_len, 1):
            att_weight = self.weight(x[:, i, :]) # (B, 1)
            out += x[:, i, :] * att_weight

        return self.relu(out + image_ori)
            
class resnet_fusion_att(nn.Module):
    def __init__(self, opt):
        super(resnet_fusion_att,self).__init__()
        self.model_name = opt.model
        self.num_classes = opt.num_classes

        if 'resnet18' in self.model_name: # (512)
            pretrained_model = models.resnet18(pretrained=True)
            self.d_model = 512
        elif 'resnet50' in self.model_name: # (2048)
            pretrained_model = models.resnet50(pretrained=True)
            self.d_model = 2048
        self.Feature_Extractor = nn.Sequential(*list(pretrained_model.children())[:-1])

        self.attention = attention(self.d_model)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.d_model, out_features=self.d_model // 2),
            nn.BatchNorm1d(self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=self.d_model // 2, out_features=self.d_model // 4),
            nn.BatchNorm1d(self.d_model // 4),
            nn.ReLU(),
            nn.Linear(in_features=self.d_model // 4, out_features=self.num_classes),
        )
    
    def forward(self, x):
        for frame_idx in range(x.shape[1]):
            frame = x[:, frame_idx, :, :, :]
            frame = self.Feature_Extractor(frame).squeeze()

            if frame.dim() == 1: # If batch_size = 1
                frame = frame.unsqueeze(0)

            frame = frame.unsqueeze(1) # pad length
            if frame_idx == 0:
                frame_out = frame
            else:
                frame_out = torch.cat((frame_out, frame), dim=1) 
        # frame_out : (B, 6, embeddings)
        out = self.attention(frame_out)
        out = self.classifier(out)

        return out

class cnn_pvt_fusion(nn.Module):
    def __init__(self, opt):
        super(cnn_pvt_fusion, self).__init__()
        self.CNN = resnet50(opt)
        self.PVT = pvt_v2_b4(True)
        self.PVT.head = nn.Linear(self.PVT.head.in_features, opt.num_classes)

        self.GAP = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(4864, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, opt.num_classes),
        )
        
    def forward(self, x):
        out_CNN, emb_CNN = self.CNN(x)
        out_PVT, emb_PVT = self.PVT(x)
        for i in range(len(emb_CNN)):
            emb_res = self.GAP(emb_CNN[i]).squeeze()
            emb_P = self.GAP(emb_PVT[i]).squeeze()
            emb_concat = torch.concat((emb_res, emb_P), dim=1)
            if i == 0:
                emb = emb_concat
            else:
                emb = torch.concat((emb, emb_concat), dim=1)
        output = self.classifier(emb)
    
        return out_CNN, out_PVT, output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='resnet18', help='resnet18/resnet50')
    parser.add_argument('--num_classes', type=int, default=15, help='number of classes')
    opt = parser.parse_args()

    x = torch.randn((2, 3, 224, 224))
    model = cnn_pvt_fusion(opt)
    
    _, _, output = model(x)
    print(output.shape)


