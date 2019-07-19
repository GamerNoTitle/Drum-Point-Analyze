import pyaudio
import wave
import numpy as np
import time
from tkinter import TclError
import matplotlib.pyplot as plt
import librosa
from scipy.fftpack import fft
import pandas as pd
import os
import sys
import json
import math
from sklearn import neighbors

filename = './features_1.json'
data = json.load(open(filename))

for path, features_dict in data.items():
    print(path)

def read_data_1():
    filename = '.\\features_1.json'
    data = json.load(open(filename))

    feature_names = sorted(list(data.values())[0])

    class_names = set()
    for path, features_dict in data.items():
        class_names.add(path.split('/')[-2])
    class_names = sorted(class_names)

    sample_names = []
    features = []
    classes = []
    for path, features_dict in data.items():
        sample_names.append(path.split('/')[-1])
        classes.append(class_names.index(path.split('/')[-2]))

        feature_vector = []
        for feature_key in feature_names:
            feature_value = features_dict[feature_key]
            if feature_value > 1000000.:
                feature_value = 1000000.
            if feature_value < -1000000.:
                feature_value = -1000000.
            if math.isnan(feature_value):
                feature_value = 0
            feature_vector.append(feature_value)

        features.append(feature_vector)
    return features, classes, sample_names, feature_names, class_names

features_1, classes_1, sample_names_1, feature_names_1, class_names_1 = read_data_1()

features = []
classes = []

for i in range(len(features_1)):
    class_id = classes_1[i]
    if class_id in [0, 3]:
        classes.append(classes_1[i])
        zcrs1 = features_1[i][3]
        cent2 = features_1[i][0]
        features.append([zcrs1, cent2])

print('features: ', features)
print('classes', classes)
print('class_name_1', class_names_1)

# %matplotlib inline

# 你需要填充以下4个数组：
tom_zcrs_max = [] # 分类为tom样本的“过零率最大值”特征值集合
tom_cent_max = [] # 分类为tom样本的“频谱质心最大值”特征值集合
closehat_zcrs_max = [] # 分类为close-hat样本的“过零率最大值”特征值集合
closehat_cent_max = [] # 分类为close-hat样本的“频谱质心最大值”特征值集合

# sample_count为样本数量
# features的shape为(sample_count, 2)，存储格式为：
# [
#     [样本1的过零率最大值，样本1的频谱质心最大值],
#     [样本2的过零率最大值，样本2的频谱质心最大值],
#     ......
# ]
# classes的shape为(sample_count, )，存储格式为：
# [
#     样本1标签，
#     样本2标签，
#     ...
# ]
# 你需要正确将每个样本的每个特征正确归类到以上四个数组中
# 例：tom_zcrs_max = [分类为tom的样本1过零率最大值, 分类为tom的样本2过零率最大值, ....]

# =========你的代码=========
print(len(features))
for i in range(len(features)):
    if (classes[i] ==3 ):
        tom_zcrs_max.append(features[i][0])
        tom_cent_max.append(features[i][1])
    else: 
        closehat_zcrs_max.append(features[i][0])
        closehat_cent_max.append(features[i][1])
print(tom_zcrs_max)
print(tom_cent_max)
print(closehat_zcrs_max)
print(closehat_cent_max)
# ========================

# 使用“过零率最大值”为x轴，“频谱质心最大值”为y轴画图

fig, ax = plt.subplots()
ax.set_title('drum wave feature visualize')
ax.set_xlabel('zero crossing rate max')
ax.set_ylabel('spectral centroid max')
ax.scatter(tom_zcrs_max,tom_cent_max,label='tom',c='b',marker='s',s=50,alpha=0.8) # 将tom样本根据特征画在散点图上
ax.scatter(closehat_zcrs_max,closehat_cent_max,label='closehat',c='r', marker='^', s=50, alpha=0.8) # 将close-hat样本根据特征画在散点
plt.legend()
plt.show()

# 构建最大近邻数为5的knn分类器
# 你需要返回以下对象：
# knn：KNN分类器对象

knn = None
# =========你的代码=========
knn = neighbors.KNeighborsClassifier() 
# ========================

# 训练knn分类器
knn.fit(features, classes)

# 使用训练集预测分类结果
# 你需要返回以下对象：
# predicted：输入为训练集特征features，在knn分类器的预测结果
predicted = None
# =========你的代码=========
predicted=knn.predict(features)
# ========================
accuracy = np.mean(predicted==classes)
print(accuracy)

CHUNK = 4 * 1024                     # 缓冲区可容纳的帧数
FORMAT = pyaudio.paInt16             # 采样数据的格式（16位int型）
CHANNELS = 1                         # 声道数
RATE = 44100                        # 麦克风采样的实际帧率（每秒采样的帧数）
RECORD_SECONDS = 0.2          # 采样秒数
CHUNK_COUNT = int(RATE / CHUNK * RECORD_SECONDS)    # 给定的采样秒数下，使用的缓冲区个数
NUM_FRAMES_IN_ALL_CHUNKS = CHUNK_COUNT * CHUNK  # 给定的采样秒数下，采集的帧数

print("* recording")

# 需要构造PyAudio类的实例，通过该实例打开流，读取一个缓冲区的采样值，关闭流
# 你需要正确生成这个对象：
# data：缓冲区内的采样数据

data = ""

# =========你的代码=========
p=pyaudio.PyAudio()
#CHUNK = 4096 #缓冲区大小
#FORMAT = pyaudio.paInt16
#CHANNELS = 2
#RATE = 44100 #采样率
#RECORD_SECONDS = 2
stream=p.open(format=FORMAT,channels=CHANNELS,rate=RATE,input=True,frames_per_buffer=CHUNK)
data=stream.read(CHUNK)
stream.stop_stream()
stream.close()
p.terminate()
# ========================

print("* done recording")

audio_data = np.frombuffer(data, dtype=np.int16)
fig, ax = plt.subplots()
ax.set_title('wave samples in a chunk')
ax.set_xlabel('frame')
ax.set_ylabel('amplitude')
ax.plot(audio_data, "-")
plt.show()

audio_data = audio_data.astype(np.float32)
# audio_data为np.array对象，使用librosa库从该段音频样本中提取过零率和频谱质心特征
# 注意：获取频谱质心函数中的采样率应为RATE
# 你需要正确生成这个对象：
# zcrs：使用librosa库从audio_data获取的过零率数据
# cent：使用librosa库从audio_data获取的频谱质心数据

zcrs = None
cent = None

# =========你的代码=========
cent=librosa.feature.spectral_centroid(y=audio_data)
zcrs=librosa.feature.zero_crossing_rate(audio_data,frame_length=CHUNK)
# ========================

print("过零率： ", zcrs)
print("频谱质心： ", cent)
# 取得过零率和频谱质心的最大值
zcrs_max = np.max(zcrs[0])
cent_max = np.max(cent[0])
print("过零率最大值：", zcrs_max)
print("频谱质心最大值：", cent_max)

predicted = knn.predict([[zcrs_max, cent_max]])
print(class_names_1[predicted[0]])

from IPython.display import display, clear_output

# %matplotlib tk

# create matplotlib figure and axes
fig, ax1 = plt.subplots(1, figsize=(15, 7))

# variable for plotting
x = np.arange(0, NUM_FRAMES_IN_ALL_CHUNKS, 1) # 样本数据

# create a line object with random data
line, = ax1.plot(x, np.random.rand(NUM_FRAMES_IN_ALL_CHUNKS), '-', lw=2)

# basic formatting for the axes
ax1.set_title('audio waves')
ax1.set_xlabel('frame')
ax1.set_ylabel('amplitude (normalized)')
ax1.set_ylim(-1, 1)
ax1.set_xlim(0, NUM_FRAMES_IN_ALL_CHUNKS)
plt.setp(ax1, xticks=[0, NUM_FRAMES_IN_ALL_CHUNKS/2, NUM_FRAMES_IN_ALL_CHUNKS], yticks=[-1, 0, 1])

# show the plot
plt.show(block=False)

audio_data = np.zeros([NUM_FRAMES_IN_ALL_CHUNKS], dtype=np.int16)
line.set_ydata(audio_data)

# %matplotlib tk

# create matplotlib figure and axes
fig, ax1 = plt.subplots(1, figsize=(15, 7))

# variable for plotting
x = np.arange(0, NUM_FRAMES_IN_ALL_CHUNKS, 1) # 样本数据

# create a line object with random data
line, = ax1.plot(x, np.random.rand(NUM_FRAMES_IN_ALL_CHUNKS), '-', lw=2)

# basic formatting for the axes
ax1.set_title('audio waves')
ax1.set_xlabel('frame')
ax1.set_ylabel('amplitude (normalized)')
ax1.set_ylim(-1024, 1024)
ax1.set_xlim(0, NUM_FRAMES_IN_ALL_CHUNKS)
plt.setp(ax1, xticks=[0, NUM_FRAMES_IN_ALL_CHUNKS/2, NUM_FRAMES_IN_ALL_CHUNKS], yticks=[-1024, 0, 1024])

# show the plot
plt.show(block=False)

audio_data = np.zeros([NUM_FRAMES_IN_ALL_CHUNKS], dtype=np.int16)
# line.set_ydata(audio_data)

# 需要构造PyAudio类的实例，和生成输入流
# 注意：需要正确设置流的缓冲区大小为CHUNK，即缓冲区内存放的帧数
# 你需要正确生成这个对象：
# p：pyaudio实例
# stream：输入流

p = None
stream = None

# =========你的代码=========
p=pyaudio.PyAudio()
stream=p.open(format=FORMAT,channels=CHANNELS,rate=RATE,input=True,frames_per_buffer=CHUNK)
#data=stream.read(CHUNK)
# ========================


display('stream started')

# 使用while循环不断从stream中读取样本，直到关闭图表位置
# 你可以在break的位置观察到：关闭matplotlib生成的图表会抛出TclError异常

while True:

    # 由于每次从流中获取数据是以缓冲区为单位，RECORD_SECONDS时间内会获取不止一个chunk大小的样本
    # 将每次从缓冲区取出的数据放入frames数组中，随后将它们拼接起来
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK, exception_on_overflow = False)
        data_np =  np.frombuffer(data, dtype=np.int16)
        frames.append(data_np)

    audio_data = np.hstack(frames)
    
    # 将样本归一化到[0, 1]范围内
    audio_data_norm = audio_data / float(np.amax(np.abs(audio_data)))
    
    
    
    # 画图
    line.set_ydata(audio_data)
    
    audio_data = audio_data.astype(np.float32)
    # audio_data包含了在设定的时间范围内获取的音频样本
    # 需要对该样本使用librosa库进行特征提取
    # 你需要正确生成这个对象：
    # zcrs：使用librosa库从audio_data获取的过零率数据
    # cent：使用librosa库从audio_data获取的频谱质心数据
    zcrs = []
    cent = []
    # =========你的代码=========
    cent=librosa.feature.spectral_centroid(y=audio_data,sr=44100)
    zcrs=librosa.feature.zero_crossing_rate(audio_data,frame_length=CHUNK)   
    # ========================
    
    # 输出结果到下方标准输出
    clear_output()
    
    display(audio_data_norm.shape)
    
    display("zcrs: " + str(zcrs))
    display("cent: " + str(cent))
    
    zcrs_max = np.max(zcrs[0]) # 过零率的最大值
    cent_max = np.max(cent[0]) # 频率质心的最大值
    display("zcrs_max: " + str(zcrs_max))
    display("cent_max: " + str(cent_max))
    
    # 使用knn分类器对上述样本的两个特征进行分类
    # 你需要正确生成这个对象：
    # predicted：使用knn分类器返回预测标签
    predicted = [None]
    # =========你的代码=========
    predicted = knn.predict([[zcrs_max, cent_max]])
    
    # ========================
    display("predict:"+ class_names_1[predicted[0]])
    
#     #  更新图表
    try:
        fig.canvas.draw()
        fig.canvas.flush_events()
    
    # 关闭图表，中断循环
    except TclError:
        display('stream stopped')
        break