import os
import numpy as np
from tqdm import tqdm
import multiprocessing
from numpy.lib.format import open_memmap

def check(dataset, modal):
    #打开待检查数据
    if dataset == 'test':
        source = open_memmap('data/{}/test.npy'.format(dataset),mode='r')
        for i in range(len(source)):
            if np.isnan(source[i]).any():
                print('TEST nan index : {} '.format(i))
    else:
        source = open_memmap('data/{}/{}.npy'.format(dataset, modal),mode='r')
        for i in range(len(source)):
            if np.isnan(source[i]).any():
                print('{}-{} nan index : {} '.format(dataset,modal,i))
    
if __name__ == '__main__':
    sets = ('train','val','test')

    modal_list = ("Bag","Hand","Hips","Torso")

    #多线程
    processes = []

    for set in sets:
        for modal in modal_list:
            process = multiprocessing.Process(target=check, args=(set, modal))
            processes.append(process)
            process.start()

    for process in processes:
        process.join()

    #单线程
    # for set in sets:
    #     for modal in modal_list:
    #         check(set, modal)