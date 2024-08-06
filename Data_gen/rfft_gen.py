import os
import torch
import numpy as np
from tqdm import tqdm
import multiprocessing
from shutil import copyfile
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from numpy.lib.format import open_memmap

def gen_rfft(dataset, modal, shape1,shape2):
    #打开原始数据
    if dataset == 'test':
        source = open_memmap('data/{}/test.npy'.format(dataset),mode='r')
        data = open_memmap('fft_data/{}/test.npy'.format(dataset),dtype=np.complex128,mode='w+',shape=shape2)
    else:
        source = open_memmap('data/{}/{}.npy'.format(dataset, modal),mode='r')
        data = open_memmap('fft_data/{}/{}.npy'.format(dataset, modal),dtype=np.complex128,mode='w+',shape=shape2)
    for i in tqdm(range(shape1[0])):
        for j in range(shape1[1]):
            data[i][j] = np.fft.rfft(source[i][j])
    
def gen_rfft_am(dataset, modal, shape1,shape2):
    #打开原始数据
    if dataset == 'test':
        source = open_memmap('data/{}/test.npy'.format(dataset),mode='r')
        data = open_memmap('fft_data/{}/test_am.npy'.format(dataset),mode='w+',shape=shape2)
    else:
        source = open_memmap('data/{}/{}.npy'.format(dataset, modal),mode='r')
        data = open_memmap('fft_data/{}/{}_am.npy'.format(dataset, modal),mode='w+',shape=shape2)
    for i in tqdm(range(shape1[0])):
        for j in range(shape1[1]):
            data[i][j] = np.abs(np.fft.rfft(source[i][j]))
            data[i][j] /= (500 / 2)
            data[i][j][0] /= 2

def img_gen(i,data,path):
    x = torch.tensor(data)

    # 设置 STFT 参数
    n_fft = 100
    hop_length = 20
    win_length = 100
    window = torch.hann_window(win_length)

    # 计算频率
    freqs = np.fft.rfftfreq(n_fft, d=1/100)

    # 计算 STFT
    stft_result = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True,normalized=True,onesided=True)

    # 计算幅度谱
    magnitude = torch.abs(stft_result)

    # 将张量转换为 numpy 数组以便使用 matplotlib
    magnitude_np = magnitude.numpy()

    magnitude_np = np.mean(magnitude_np, axis=0)

    # 频率和时间轴
    times = np.arange(magnitude_np.shape[1]) * hop_length / 1024

    # 创建图像和轴对象
    fig = plt.figure(figsize=(1.28, 1.28), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.pcolormesh(times, freqs, magnitude_np, shading='gouraud')

    # 隐藏坐标轴
    ax.set_axis_off()

    # 保存频谱图
    fig.savefig(os.path.join(path,str(i) + ".jpg"), bbox_inches='tight', pad_inches=0)
    # 关闭图像以释放内存
    plt.close(fig)

def am_img_gen(dataset, modal):
    if dataset == 'test' and modal != 'Bag':
        return
    if dataset == 'test':
        modal = ''
    if not os.path.exists(os.path.join('img_data')):
        os.mkdir(os.path.join('img_data'))
    if not os.path.exists(os.path.join('img_data',dataset)):
        os.mkdir(os.path.join('img_data',dataset))
    if not os.path.exists(os.path.join('img_data',dataset,modal)):
        os.mkdir(os.path.join('img_data',dataset,modal))

    if dataset == 'test':
        print("processing {}".format(dataset))
        source = np.load('data/{}/test.npy'.format(dataset),mmap_mode='r')
        path = os.path.join('img_data',dataset)
    else:
        print("processing {} {}".format(dataset,modal))
        source = np.load('data/{}/{}.npy'.format(dataset, modal),mmap_mode='r')
        path = os.path.join('img_data',dataset,modal)
    Parallel(n_jobs=8)(delayed(img_gen)(i,source[i],path) for i in tqdm(range(source.shape[0])))

if __name__ == '__main__':
    
    #形状对应：帧，数据源(九个通道代表Acc、Gyr、Mag的各xyz通道)，500个样本
    shape_data_list = [(196072, 9, 500),(28789, 9, 500),(92726, 9, 500)]

    shape_fft_list = [(196072, 9, 251),(28789, 9, 251),(92726, 9, 251)]

    shape_image_list = [(196072, 3, 256,256),(28789, 3, 256,256),(92726, 3, 256,256)]

    sets = ('train', 'val','test')

    modal_list = ("Bag","Hand","Hips","Torso")

    path=os.path.join('fft_data')
    if not os.path.exists(path):
        os.mkdir(path)
    for dataset in sets:
        path=os.path.join('fft_data',dataset)
        if not os.path.exists(path):
            os.mkdir(path)
    for i in range(0,2):
        copyfile('data/{}/label.npy'.format(sets[i]),'fft_data/{}/label.npy'.format(sets[i]))

    #多线程
    processes = []

    for set in sets:
        for modal in modal_list:
            process = multiprocessing.Process(target=gen_rfft_am, args=(set, modal,shape_data_list[sets.index(set)],shape_fft_list[sets.index(set)]))
            processes.append(process)
            process.start()
    for process in processes:
        process.join()

    # 单线程
    # for set in sets:
    #     for modal in modal_list:
    #         gen_rfft_am(set, modal,shape_data_list[sets.index(set)],shape_fft_list[sets.index(set)])
    #         am_img_gen(set, modal)