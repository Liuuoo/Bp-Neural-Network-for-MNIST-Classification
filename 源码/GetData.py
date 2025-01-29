from tensorflow.keras.datasets import mnist
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # 1：仅显示警告，2：仅显示错误，3：隐藏所有日志

class GetData:
    def __init__(self, x_train=None, y_train=None, x_test=None, y_test=None, x_val=None, y_val=None):
        self.x_train = x_train if x_train is not None else []
        self.y_train = y_train if y_train is not None else []
        self.x_test = x_test if x_test is not None else []
        self.y_test = y_test if y_test is not None else []
        self.x_val = x_val if x_val is not None else []
        self.y_val = y_val if y_val is not None else []

    # 加载 MNIST 数据集并划分验证集
    @staticmethod
    def create_dataset(val_ratio=0.1):
        """
        加载 MNIST 数据集，并划分训练集、验证集和测试集。

        参数:
            val_ratio (float): 验证集的比例，默认为 10%。

        返回:
            GetData: 包含训练集、验证集和测试集的实例。
        """
        # 加载 MNIST 数据集
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # 数据归一化，将像素值从 [0, 255] 映射到 [0, 1]
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

        # 将二维图像展平为一维数组
        x_train = x_train.reshape(x_train.shape[0], 784, 1)
        x_test = x_test.reshape(x_test.shape[0], 784, 1)

        # 划分验证集
        val_size = int(val_ratio * len(x_train))  # 计算验证集大小
        indices = np.arange(len(x_train))  # 生成索引
        np.random.shuffle(indices)  # 打乱索引

        # 划分训练集和验证集
        x_val = x_train[indices[:val_size]]
        y_val = y_train[indices[:val_size]]
        x_train = x_train[indices[val_size:]]
        y_train = y_train[indices[val_size:]]

        # 返回包含训练集、验证集和测试集的实例
        return GetData(x_train, y_train, x_test, y_test, x_val, y_val)