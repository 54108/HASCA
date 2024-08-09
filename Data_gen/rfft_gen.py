import os
import argparse
import cupy as cp
import numpy as np
from tqdm import tqdm
import multiprocessing
from shutil import copyfile
from numpy.lib.format import open_memmap

def get_available_memory():
    mem_info = cp.cuda.runtime.memGetInfo()
    return mem_info[0]  # 返回可用显存大小

def calculate_chunk_size(available_memory, shape1, dtype_size=8):
    # 估算每个样本的内存占用
    sample_size = shape1[1] * shape1[2] * dtype_size
    # 估算每个块的内存占用，包括原始数据和傅里叶变换结果
    chunk_size = available_memory // ((6) * sample_size)
    return max(1, int(chunk_size))
    
def gen_rfft(dataset, modal, shape1,shape2):
    #打开原始数据
    if dataset == 'test':
        source = open_memmap('data/{}/test.npy'.format(dataset),mode='r')
        data = open_memmap('fft_data/{}/test.npy'.format(dataset),mode='w+',dtype=np.float32,shape=shape2)
    else:
        source = open_memmap('data/{}/{}.npy'.format(dataset, modal),mode='r')
        data = open_memmap('fft_data/{}/{}.npy'.format(dataset, modal),mode='w+',dtype=np.float32,shape=shape2)
    for i in tqdm(range(shape1[0])):
        for j in range(shape1[1]):
            data[i][j] = np.abs(np.fft.rfft(source[i][j]))
            data[i][j] /= (500 / 2)
            data[i][j][0] /= 2

def gen_rfft_cu(dataset, modal, shape1,shape2):
    #打开原始数据
    if dataset == 'test':
        source = open_memmap('data/{}/test.npy'.format(dataset),mode='r')
        data = open_memmap('fft_data/{}/test_cu.npy'.format(dataset),mode='w+',dtype=np.float32,shape=shape2)
    else:
        source = open_memmap('data/{}/{}.npy'.format(dataset, modal),mode='r')
        data = open_memmap('fft_data/{}/{}_cu.npy'.format(dataset, modal),mode='w+',dtype=np.float32,shape=shape2)
    # 分块处理
    num_chunks = (shape1[0] + chunk_size - 1) // chunk_size
    gpu_data = cp.zeros((chunk_size, shape1[1], shape1[2]))
    for chunk_idx in tqdm(range(num_chunks)):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, shape1[0])
        gpu_data = cp.reshape(gpu_data, (((end_idx-start_idx)*shape1[1]), shape1[2]))
        gpu_result = cp.fft.rfft(gpu_data)
        gpu_result = cp.reshape(gpu_result, (end_idx-start_idx, shape1[1], shape1[2]//2+1))
        # 将结果从 GPU 内存复制回 CPU 内存
        data[start_idx:end_idx] = cp.asnumpy(cp.abs(gpu_result))
    data /= (500 / 2)
    data[:,:,0] /= 2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate RFFT amplitude data.')
    parser.add_argument('--cupy', action='store_true', help='Use CuPy for GPU acceleration')
    args = parser.parse_args()

    #形状对应：帧，数据源(九个通道代表Acc、Gyr、Mag的各xyz通道)，500个样本
    shape_data_list = [(196072, 9, 500),(28789, 9, 500),(92726, 9, 500)]

    shape_fft_list = [(196072, 9, 251),(28789, 9, 251),(92726, 9, 251)]

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

    if args.cupy:
        # 单线程
        for set in sets:
            for modal in modal_list:
                # 自动设置 chunk_size
                available_memory = get_available_memory()
                if available_memory:
                    index = sets.index(set)
                    chunk_size = calculate_chunk_size(available_memory, shape_data_list[index])
                    print(f"Using chunk size: {chunk_size}")
                else:
                    chunk_size = 1000  # 默认值
                gen_rfft_cu(set, modal,shape_data_list[sets.index(set)],shape_fft_list[sets.index(set)])
    else:
        #多线程
        processes = []

        for set in sets:
            for modal in modal_list:
                process = multiprocessing.Process(target=gen_rfft, args=(set, modal,shape_data_list[sets.index(set)],shape_fft_list[sets.index(set)]))
                processes.append(process)
                process.start()
        for process in processes:
            process.join()

    