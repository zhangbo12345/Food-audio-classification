# -*- coding: utf-8 -*-

import os
import glob

import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display

from tqdm import tqdm

feature = []
label = []
# 建立类别标签，不同类别对应不同的数字。
label_dict = {'aloe': 0, 'burger': 1, 'cabbage': 2, 'candied_fruits': 3, 'carrots': 4, 'chips': 5,
              'chocolate': 6, 'drinks': 7, 'fries': 8, 'grapes': 9, 'gummies': 10, 'ice-cream': 11,
              'jelly': 12, 'noodles': 13, 'pickles': 14, 'pizza': 15, 'ribs': 16, 'salmon': 17,
              'soup': 18, 'wings': 19}
label_dict_inv = {v: k for k, v in label_dict.items()}

def extract_features(parent_dir, sub_dirs, max_file=10, file_ext="*.wav"):
    c = 0
    label, feature = [], []
    for sub_dir in sub_dirs:
        for fn in tqdm(glob.glob(os.path.join(parent_dir, sub_dir, file_ext))):  # 遍历数据集的所有文件
            # print(glob.glob(os.path.join(parent_dir, sub_dir, file_ext))[:max_file])
            # segment_log_specgrams, segment_labels = [], []
            # sound_clip,sr = librosa.load(fn)
            # print(fn)
            label_name = fn.split('/')[-2]  # 类别（文件夹名字）
            # print(label_name)
            # label_name = label_name.split('.')[-2]
            # print(label_name)
            label.extend([label_dict[label_name]])
            X, sample_rate = librosa.load(fn, res_type='kaiser_fast')
            mel = librosa.feature.melspectrogram(y=X, sr=sample_rate)
            mels = np.mean(librosa.power_to_db(mel).T, axis=0)  # 计算梅尔频谱(mel spectrogram),并把它作为特征
            # mels = np.mean(librosa.feature.melspectrogram(y=X,sr=sample_rate).T,axis=0) # 计算梅尔频谱(mel spectrogram),并把它作为特征
            feature.extend([mels])

    return [feature, label]

def get_npy(parent_dir = './train'):


    # 自己更改目录
    save_dir = "./"
    folds = sub_dirs = np.array(['aloe', 'burger', 'cabbage', 'candied_fruits',
                                 'carrots', 'chips', 'chocolate', 'drinks', 'fries',
                                 'grapes', 'gummies', 'ice-cream', 'jelly', 'noodles', 'pickles',
                                 'pizza', 'ribs', 'salmon', 'soup', 'wings'])

    # 获取特征feature以及类别的label
    temp = extract_features(parent_dir, sub_dirs)
    temp = np.array(temp)
    data = temp.transpose()
    np.save('./train.npy', data)


def get_logmfcc_pic(audio='./test.wav'):
    X, sample_rate = librosa.load(audio, res_type='kaiser_fast')
    mel = librosa.feature.melspectrogram(y=X, sr=sample_rate)
    mels = np.mean(librosa.power_to_db(mel).T, axis=0)  # 计算梅尔频谱(mel spectrogram),并把它作为特征
    pic = mels.reshape(16,8)
    plt.imshow(pic)
    plt.savefig('./feature.png',dpi = 300)


if __name__ == '__main__':
    # get_npy()
    get_logmfcc_pic()
