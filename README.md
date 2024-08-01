## 说明
### 数据应当按照如下格式组织

```
.
├── Data
│   ├── data_process.ipynb
│   ├── SHL-2024-Test
│   │   ├── Acc_x.txt
│   │   └── ...
│   ├── SHL-2024-Train_Bag
│   │   └── train
│   │       └── Bag
│   │           ├── Acc_x.txt
│   │           └── ...
│   ├── SHL-2024-Train_Hand
│   │   └── train
│   │       └── Hand
│   │           ├── Acc_x.txt
│   │           └── ...
│   ├── SHL-2024-Train_Hips
│   │   └── train
│   │       └── Hips
│   │           ├── Acc_x.txt
│   │           └── ...
│   ├── SHL-2024-Train_Torso
│   │   └── train
│   │       └── Torso
│   │           ├── Acc_x.txt
│   │           └── ...
│   ├── SHL-2024-Validation
│   │   └── validation
│   │       ├── Bag
│   │       │   ├── Acc_x.txt
│   │       │   └── ...
│   │       ├── Hand
│   │       │   ├── Acc_x.txt
│   │       │   └── ...
│   │       ├── Hips
│   │       │   ├── Acc_x.txt
│   │       │   └── ...
│   │       └── Torso
│   │           ├── Acc_x.txt
│   │           └── ...
│   └── SimHei.ttf
└── README.md
```
## Data目录下data_process.ipynb傅里叶相关部分应当运行仅一次，防止不必要的生成。值得注意的是对label.npy中的数据进行傅里叶变换是毫无意义的。