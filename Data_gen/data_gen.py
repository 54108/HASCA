import os
import numpy as np
import multiprocessing
from tqdm import tqdm
from numpy.lib.format import open_memmap

def gen_data(dataset, modal, file_list, shape):
    #test不存在多模态
    if dataset == 'test':
        modal = ''
    #生成实际file位置
    file_list = list(file_list)
    for i in range(len(file_list)):
        file_list[i] = os.path.join('raw_data',dataset,modal,file_list[i])
    #生成数据
    if dataset == 'test':
        print("Processing test")
        # data = open_memmap('data/{}/test.npy'.format(dataset),dtype=np.float32,mode='w+',shape=shape)
        data = open_memmap('data/{}/test.npy'.format(dataset),mode='w+',shape=shape)
    else:
        print("Processing {} {}".format(dataset,modal))
        # data = open_memmap('data/{}/{}.npy'.format(dataset, modal),dtype=np.float32,mode='w+',shape=shape)
        data = open_memmap('data/{}/{}.npy'.format(dataset, modal),mode='w+',shape=shape)
    for i in tqdm(range(9)):
        if dataset == 'train':
            data[:, i, :] = np.loadtxt(file_list[i])
        else:
            data[:, i, :] = np.loadtxt(file_list[i],delimiter=',')
    # process some nan 
    if dataset == 'train' and modal == 'Hips':
        data[121217] = np.nan_to_num(data[121217],copy=False,nan=0.005)

def gen_label(dataset, modal,label_file):
    if dataset == 'test':
        return
    source_path = os.path.join('raw_data',dataset,modal,label_file)
    #生成label
    label = np.loadtxt(source_path, dtype=np.int8)
    np.save('data/{}/label.npy'.format(dataset),label)

if __name__ == '__main__':

    #形状对应：帧，数据源(九个通道代表Acc、Gyr、Mag的各xyz通道)，500个样本
    shape_list = [(196072, 9, 500),(28789, 9, 500),(92726, 9, 500)]

    sets = ('train', 'val','test')

    modal_list = ("Bag","Hand","Hips","Torso")

    file_list = ("Acc_x.txt", "Acc_y.txt", "Acc_z.txt","Gyr_x.txt", "Gyr_y.txt", "Gyr_z.txt","Mag_x.txt", "Mag_y.txt", "Mag_z.txt")

    label_file = "Label.txt"

    if not os.path.exists('data'):
        os.mkdir('data')
    for dataset in sets:
        path=os.path.join('data',dataset)
        if not os.path.exists(path):
            os.mkdir(path)

    for i in range(0,2):
        gen_label(sets[i],modal_list[0],label_file)

    # 多线程
    processes = []

    for set in sets:
        process = multiprocessing.Process(target=gen_label, args=(set, modal_list[0],label_file))
        processes.append(process)
        for modal in modal_list:
            if set == 'test' and modal == 'Bag' :
                process = multiprocessing.Process(target=gen_data, args=(set, modal_list[0],file_list,shape_list[sets.index(set)]))
                processes.append(process)
            elif set == 'train' or set == 'val':
                process = multiprocessing.Process(target=gen_data, args=(set, modal,file_list,shape_list[sets.index(set)]))
                processes.append(process)

    for process in processes:
        process.start()
    for process in processes:
        process.join()

    #单线程
    # for set in sets:
    #     gen_label(set, modal_list[0],label_file)
    #     for modal in modal_list:
    #         if set == 'test':
    #             gen_data(set, modal_list[0],file_list,shape_list[sets.index(set)])
    #         else:
    #             gen_data(set, modal,file_list,shape_list[sets.index(set)])