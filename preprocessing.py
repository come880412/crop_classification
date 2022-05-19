import numpy as np
import cv2
from PIL import Image
import os
import tqdm

import torchvision.transforms as transforms

np.random.seed(2022)
os.environ['PYTHONHASHSEED'] = str(2022)

def train_val_split(data, split_ratio, data_info, folder_name):
    data_len = len(data)

    index = np.random.choice(data_len, data_len, replace=False)
    train_index = index[:int(data_len * split_ratio[0])]
    valid_index = index[int(data_len * split_ratio[0]): int(data_len * (split_ratio[0] + split_ratio[1]))]
    test_index = index[int(data_len * (split_ratio[0] + split_ratio[1])):]

    train_data = data[train_index]
    valid_data = data[valid_index]
    test_data = data[test_index]

    data_info[folder_name].append(len(train_data))
    data_info[folder_name].append(len(valid_data))
    data_info[folder_name].append(len(test_data))

    return train_data, valid_data, test_data

if __name__ == "__main__":
    transform  =   transforms.Compose([
                    transforms.Resize((1024, 1024))])
    data_path = '../dataset/original'
    split_ratio = [0.9,0.1,0] # train/valid/test

    image_size = 1024
    save_image_path = '../dataset/image_size%d' % image_size
    save_file_path = '../dataset'
    
    ignore_file = ['train.txt', 'valid.txt', 'test.txt']

    data_info = {}
    count = 0
    for folder_name in tqdm.tqdm(os.listdir(data_path)):
        if folder_name in ignore_file:
            continue
        os.makedirs(os.path.join(save_image_path, folder_name), exist_ok=True)
        data_info[folder_name] = []
        temp_data = []

        for image_name in os.listdir(os.path.join(data_path, folder_name))[:10]:
            image_path = os.path.join(data_path, folder_name, image_name)

            image = Image.open(image_path).convert('RGB')
            image = transform(image)

            image.save(os.path.join(save_image_path, folder_name, image_name[:-4] + '.png'))
            temp_data.append(os.path.join(folder_name, image_name[:-4] + '.png'))
        temp_data = np.array(temp_data)

        if count == 0:
            train_data, valid_data, test_data = train_val_split(temp_data, split_ratio, data_info, folder_name)
        else:
            temp_train, temp_valid, temp_test = train_val_split(temp_data, split_ratio, data_info, folder_name)
            train_data = np.concatenate((train_data, temp_train))
            valid_data = np.concatenate((valid_data, temp_valid))
            test_data = np.concatenate((test_data, temp_test))
        count += 1
    
    np.savetxt(os.path.join(save_file_path, 'train.txt'),  train_data, fmt='%s', delimiter=',')
    np.savetxt(os.path.join(save_file_path, 'valid.txt'),  valid_data, fmt='%s', delimiter=',')
    np.savetxt(os.path.join(save_file_path, 'test.txt'),  test_data, fmt='%s', delimiter=',')

    print('---------------Statistics---------------')
    print('# Training data / # Validation data / # testing data')
    for key, value in data_info.items():
        print('%s : %d / %d / %d' % (key, value[0], value[1], value[2]))    
        
