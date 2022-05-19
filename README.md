# Crop_classification
Competition for 2022 AI Cup "農地作物現況調查影像辨識競賽 - 春季賽：AI作物影像判釋" (3rd place) \
Competition URL: https://aidea-web.tw/topic/93c8c26b-0e96-44bc-9a53-1c96353ad340

# Getting started
- Clone this repo to your local
``` bash
git clone https://github.com/come880412/crop_classification
cd crop_classification
```

# Data preprocessing
First, download the dataset from the [official](https://aidea-web.tw/topic/93c8c26b-0e96-44bc-9a53-1c96353ad340). And put all the data into a director, named `../dataset/Original`. Then use the following command to resize the images into 1024 * 1024 and split the data into train/valid sets. 
``` bash
python preprocessing.py
```

# Inference
First, download the pretrained models from [here](https://drive.google.com/drive/folders/1pV7l9Gf5WBrCAbtnW9h5WB_HKJ17jr3c?usp=sharing). And put the models into the director `./checkpoints`. Then use the following command to do model inference (You may need to change the dataset path on your own).
``` bash
python test.py --root path/to/public_data
```
- In the public data folder, it should contain the directory `images` and the file `submission_example.csv`.

# Training
After preparing the data by those mentioned above, you could use the script `train.sh` to train the model from scratch. Please see more detail in this script if you want to train your model.
``` bash
bash train.sh
```

# Grad-cam visualization
After training, we used grad cam to visualize where the model focuses. The visualization results are shown below.
<img src="https://github.com/come880412/crop_classification/blob/main/images/20171129-1-0165.jpg" width=41% height=41%>|<img src="https://github.com/come880412/crop_classification/blob/main/images/20180626-3-0028.jpg" width=40% height=40%>
<img src="https://github.com/come880412/crop_classification/blob/main/images/160118-3-0086.jpg" width=41% height=41%>|<img src="https://github.com/come880412/crop_classification/blob/main/images/20170205-1-0021.jpg" width=40% height=40%>

These results show that our model learns the most important features in the corresponding class, instead of overfitting some unimportant features.
