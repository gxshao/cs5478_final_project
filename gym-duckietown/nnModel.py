import os
import cv2
import random
import math
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from torchvision import transforms
from torchvision.transforms import *
import sys
import torch.nn as nn
import torch.nn.functional as F


# class Flatten(nn.Module):
#     def forward(self, x):
#         return x.view(x.size(0), -1)

class MyModel(nn.Module):

    def __init__(self, num_bins=5):
        super().__init__()
        self.num_bins = num_bins

        # Build the CNN feature extractor
        self.cnn = nn.Sequential(

            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.05),

            nn.AdaptiveMaxPool2d(output_size=(1, 1))
        )

        # Build a FC heads, taking both the image features and the intention as input 
        self.fc = nn.Sequential(
                    nn.Linear(in_features=192+3, out_features=32),
                    nn.Linear(in_features=32, out_features=num_bins))


    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return x


def read_image(path):
    return Image.open(path)

def predict(im):
    my_model = MyModel()
    my_model = nn.DataParallel(my_model.cuda().float())
    my_model.eval()

    my_model.load_state_dict(torch.load('../model/model_update.pt'))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    my_model = my_model.to(device)


    preprocessor = Compose([
            Resize((160, 160)),
            ToTensor(),
            Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2675, 0.2565, 0.2761])
        ])

    image = preprocessor(im)[None, ...]

    with torch.no_grad():
        prediction = my_model(image)
        _,score = torch.max(prediction, 1)
        return score

    # for i, image in enumerate(test_data.__getitem__()[None, ...]):
        # image = image.cuda()
        # with torch.no_grad():
        #     prediction = my_model(image)
        #     print(prediction)

if __name__ == "__main__":
    for i in range(83):
        img = read_image('../result/images/{}.png'.format(i))
        print('test predict function : '+str(predict(img).item()))