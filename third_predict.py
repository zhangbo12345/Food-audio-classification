import os
import glob
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
from tqdm import tqdm
from first_get_feature import label_dict_inv

def get_pre(audio = './test.wav'):
    class MyConvNet(nn.Module):
        def __init__(self):
            super(MyConvNet, self).__init__()
            ## 定义第一个卷积层
            self.conv1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=1,  ## 输入的feature map
                    out_channels=64,  ## 输出的feature map
                    kernel_size=3,  ##卷积核尺寸
                    stride=1,  ##卷积核步长
                    padding=5,  # 进行填充
                ),  ## 卷积后： (1*28*28) ->(16*28*28)
                nn.Tanh(),  # 激活函数
                nn.AvgPool2d(
                    kernel_size=2,  ## 平均值池化层,使用 2*2
                    stride=1,  ## 池化步长为2
                ),  ## 池化后：(16*28*28)->(16*14*14)
            )
            ## 定义第二个卷积层
            self.conv2 = nn.Sequential(
                nn.Conv2d(64, 128, 3, 1, 1),  ## 卷积操作(16*14*14)->(32*12*12)
                nn.Tanh(),  # 激活函数
                nn.AvgPool2d(2, 1)  ## 最大值池化操作(32*12*12)->(32*6*6)
            )
            ## 定义第二个卷积层
            self.conv3 = nn.Sequential(
                nn.Conv2d(128, 256, 3, 1, 1),  ## 卷积操作(16*14*14)->(32*12*12)
                nn.Tanh(),  # 激活函数
                nn.AvgPool2d(2, 1)  ## 最大值池化操作(32*12*12)->(32*6*6)
            )
            self.conv4 = nn.Sequential(
                nn.Conv2d(256, 128, 3, 1, 1),  ## 卷积操作(16*14*14)->(32*12*12)
                nn.Tanh(),  # 激活函数
                nn.AvgPool2d(2, 1)  ## 最大值池化操作(32*12*12)->(32*6*6)
            )
            self.conv5 = nn.Sequential(
                nn.Conv2d(128, 64, 3, 1, 1),  ## 卷积操作(16*14*14)->(32*12*12)
                nn.Tanh(),  # 激活函数
                nn.AvgPool2d(2, 1)  ## 最大值池化操作(32*12*12)->(32*6*6)
            )
            self.classifier = nn.Sequential(
                nn.Linear(13376, 5012),
                nn.Tanh(),
                nn.Dropout(p=0.5),
                nn.Linear(5012, 1024),
                nn.Tanh(),
                nn.Dropout(p=0.5),
                nn.Linear(1024, 256),
                nn.Tanh(),
                nn.Dropout(p=0.5),
                nn.Linear(256, 20),
            )
            self.dropout_c = nn.Dropout(p=0.5)  # dropout训练

        ## 定义网络的向前传播路径
        def forward(self, x):
            x = self.conv1(x)
            x = self.dropout_c(x)
            x = self.conv2(x)
            x = self.dropout_c(x)
            x = self.conv3(x)
            x = self.dropout_c(x)
            x = self.conv4(x)
            x = self.dropout_c(x)
            x = self.conv5(x)
            x = self.dropout_c(x)
            x = x.view(x.size(0), -1)  # 展平多维的卷积图层
            # print(x.shape)
            output = self.classifier(x)
            return output

    model = MyConvNet()
    model.load_state_dict(torch.load("./model/cnn.pkl"))
    X, sample_rate = librosa.load(audio, res_type='kaiser_fast')
    mel = librosa.feature.melspectrogram(y=X, sr=sample_rate)
    mel = np.mean(librosa.power_to_db(mel).T, axis=0)  # 计算梅尔频谱(mel spectrogram),并把它作为特征
    X_test = torch.tensor(np.vstack(mel))
    predictions = model(X_test.reshape(-1, 1, 16, 8))
    result = label_dict_inv[np.argmax(predictions.detach().numpy())]
    print(result)
    print('预测完毕')
    return result

if __name__ == '__main__' :
    get_pre()

