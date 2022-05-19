from tkinter import image_names
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch


import os

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import math

data_dict = {'banana':0, 'bareland':1, 'carrot':2, 'corn':3, 'dragonfruit':4, 'garlic':5,
             'guava':6, 'peanut':7, 'pineapple':8, 'pumpkin':9, 'rice':10,
             'soybean':11, 'sugarcane':12, 'tomato':13}

transforms_test =   transforms.Compose([
                    transforms.Resize((1024, 1024)),
                    transforms.ToTensor()])

transforms_train =  transforms.Compose([
                    transforms.Resize((1024, 1024)),
                    transforms.RandomRotation(10),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    transforms.ToTensor(),])

class Crop_data(Dataset):
    def __init__(self, opt, root, mode):
        self.root = root
        self.mode = mode
        self.is_five_crop = opt.five_crop

        self.data = []
        self.data_name = []

        if mode == 'train':
            self.transform = transforms_train
        elif mode == 'valid':
            self.transform = transforms_test  
        elif mode == 'test':       
            self.transform = transforms_test    
            data = np.loadtxt(os.path.join(self.root, 'submission_example.csv'), delimiter=',', dtype=np.str)[1:]

        if mode == 'train' or mode == 'valid':
            with open(os.path.join(self.root, '%s.txt' % self.mode), 'r') as file:
                line = file.read().splitlines()
                for info in line:
                    class_name = info.split('/')[0] # '/' for linux os
                    data_name = info.split('/')[1]

                    image_path = os.path.join(self.root, class_name, data_name[:-4] + '.jpg')
                    label = data_dict[class_name]

                    self.data.append([image_path, label])
                    self.data_name.append(data_name[:-4] + '.jpg')
        elif mode == 'test':
            for info in data:
                image_name, _ = info
                image_path = os.path.join(self.root, 'images', image_name)
                self.data.append([image_path, image_name])

    def __getitem__(self,index):
        if self.mode == 'train' or self.mode == 'valid':
            image_path, label = self.data[index]
            if self.is_five_crop:
                image = cv2.imread(image_path, 1)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                five_crop_image = self.five_crop(image)
                for idx, cropped_image in enumerate(five_crop_image):
                    if idx == 0:
                        transformed_image = self.transform(Image.fromarray(cropped_image)).unsqueeze(0)
                    else:
                        temp_tensor = self.transform(Image.fromarray(cropped_image)).unsqueeze(0)
                        transformed_image = torch.cat((transformed_image, temp_tensor), dim=0)

                return transformed_image, label, self.data_name[index]
            
            else:
                image = Image.open(image_path).convert('RGB')
                return self.transform(image), label, self.data_name[index]
        elif self.mode == 'test':
            image_path, image_name = self.data[index]
            image = Image.open(image_path).convert('RGB')
            return self.transform(image), image_name
    
    def five_crop(self, image):
        h, w, _ = image.shape
        center_h, center_w = h // 2, w // 2

        left_up_corner = image[:center_h, :center_w, :]
        left_down_corner = image[center_h:, :center_w, :]
        right_up_corner = image[:center_h, center_w:, :]
        right_down_corner = image[center_h:, center_w:, :]
        center = image[center_h - center_h//2 : center_h + center_h // 2, center_w - center_w // 2 : center_w + center_w // 2, :]
        image_list = np.array([image, left_up_corner, left_down_corner, right_up_corner, right_down_corner, center])

        # show images
        # label_list = ['Original image', 'left_up', 'left_down', 'right_up', 'right_down', 'center_crop']
        # for i in range(len(image_list)):
        #     img = image_list[i]
        #     plt.subplot(3, 2, i+1)
        #     plt.title(label_list[i])
        #     plt.imshow(img)
        # plt.show()
        
        return image_list
        
    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    dataset = Crop_data('../dataset', 'train')
    train_loader = DataLoader(dataset, batch_size=2,shuffle=True, num_workers=0)
    iter_data = iter(train_loader)
    image, label = iter_data.next()
    
    print(image.shape, label)