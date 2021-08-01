# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
import torch.utils.data as Data
from torchvision import transforms
import torch.nn as nn
import copy
import time

from tqdm import tqdm



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

def get_model():

    data = np.load('./train.npy', allow_pickle=True)

    # 获取特征
    X = np.vstack(data[:, 0])

    # 获取标签
    Y = np.array(data[:, 1])
    Y = np.asarray(Y, 'int64')
    print()
    print('X的特征尺寸是：', X.shape)
    print('Y的特征尺寸是：', Y.shape)


    # 设置种子
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)


    # 设置随机数种子
    setup_seed(2021)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1, test_size=0.10)
    print('训练集的大小', len(X_train))
    print('测试集的大小', len(X_test))

    X_train = X_train.reshape(-1, 1, 16, 8)
    Y_train = np.asarray(Y_train, 'int64')
    X_test = X_test.reshape(-1, 1, 16, 8)

    X_train = torch.tensor(X_train)
    Y_train = torch.tensor(Y_train, dtype=torch.int64)
    train_data = Data.TensorDataset(X_train, Y_train)
    X_test = torch.tensor(X_test)
    Y_test = torch.tensor(Y_test, dtype=torch.int64)
    test_data = Data.TensorDataset(X_test, Y_test)

    bs = 120
    # 定义一个数据加载器
    train_loader = Data.DataLoader(
        dataset=train_data,  # 使用的数据集
        batch_size=bs,  # 批处理样本大小
        shuffle=True,  # 每次迭代前打乱数据
    )

    test_loader = Data.DataLoader(
        dataset=test_data,  # 使用的数据集
        batch_size=bs,  # 批处理样本大小
        shuffle=True,  # 每次迭代前打乱数据
    )

    for step, (b_x, b_y) in enumerate(train_loader):
        if step > 0:
            break

    print(b_x.shape)
    print(b_y.shape)
    print(b_x.dtype)
    print(b_y.dtype)
    # 可视化一个batch的图像
    batch_x = b_x.squeeze().numpy()
    batch_y = b_y.numpy()
    plt.figure(figsize=(12, 5))
    for ii in np.arange(20):
        plt.subplot(4, 5, ii + 1)
        plt.imshow(batch_x[ii, :, :], cmap=plt.cm.gray)
        plt.title(batch_y[ii], size=9)
        plt.axis("off")
    plt.show()

    model = MyConvNet()

    device = torch.device('cuda')
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
        model.to(device)
    elif torch.cuda.device_count() == 1:
        model.to(device)
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    else:
        print('no gpu can use')
    print(model)

    ## 对模型进行训练
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    criterion = nn.CrossEntropyLoss()  # 损失函数
    num_epochs = 30

    ## 计算训练使用的batch数量
    batch_num = len(train_loader)
    ## 复制模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    since = time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # 每个epoch有两个训练阶段
        train_loss = 0.0
        train_corrects = 0
        train_num = 0
        val_loss = 0.0
        val_corrects = 0
        val_num = 0

        # 训练部分
        for step, (b_x, b_y) in enumerate(train_loader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            model.train()  ## 设置模型为训练模式
            optimizer.zero_grad()
            output = model(b_x)
            pre_lab = torch.argmax(output, 1)
            loss = criterion(output, b_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * b_x.size(0)
            train_corrects += torch.sum(pre_lab == b_y.data)
            train_num += b_x.size(0)

        # 测试部分
        for step, (b_x, b_y) in enumerate(test_loader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            model.eval()  ## 设置模型为训练模式评估模式
            output = model(b_x)
            pre_lab = torch.argmax(output, 1)
            loss = criterion(output, b_y)
            val_loss += loss.item() * b_x.size(0)
            val_corrects += torch.sum(pre_lab == b_y.data)
            val_num += b_x.size(0)
        ## 计算一个epoch在训练集和验证集上的的损失和精度
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)
        print('{} Train Loss: {:.4f}  Train Acc: {:.4f}'.format(
            epoch, train_loss_all[-1], train_acc_all[-1]))
        print('{} Val Loss: {:.4f}  val Acc: {:.4f}'.format(
            epoch, val_loss_all[-1], val_acc_all[-1]))
        # 拷贝模型最高精度下的参数
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
        time_use = time.time() - since
        print("Train and val complete in {:.0f}m {:.0f}s".format(
            time_use // 60, time_use % 60))
    # 使用最好模型的参数
    model.load_state_dict(best_model_wts)

    # 检测是否存在保存目录,不存在就生成
    if not os.path.exists('./model'):
        os.mkdir('./model')

    torch.save(model.state_dict(), './model/cnn.pkl')
    train_process = pd.DataFrame(
        data={"epoch": range(num_epochs),
              "train_loss_all": train_loss_all,
              "val_loss_all": val_loss_all,
              "train_acc_all": train_acc_all,
              "val_acc_all": val_acc_all})

    ## 可视化模型训练过程
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process.epoch, train_process.train_loss_all,
             "r-", label="Train loss")
    plt.plot(train_process.epoch, train_process.val_loss_all,
             "b-", label="Val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(train_process.epoch, train_process.train_acc_all,
             "r-", label="Train acc")
    plt.plot(train_process.epoch, train_process.val_acc_all,
             "b-", label="Val acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.grid()
    plt.savefig("filename.png",dpi = 300)
    plt.show()

if __name__ == "__main__":
    get_model()