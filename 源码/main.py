import numpy as np
from BackCalc import calc_weight
from GetData import GetData

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class BpNetwork:
    def __init__(self, learn_alpha, hidden_size, input_size=784, output_size=10,
                 front_edge_file="FrontEdge.txt", back_edge_file="BackEdge.txt"):
        """
        初始化神经网络
        :param learn_alpha: 学习率
        :param input_size: 输入层神经元数量（默认为784）
        :param hidden_size: 隐藏层神经元数量（默认为100）
        :param output_size: 输出层神经元数量（默认为10）
        :param front_edge_file: 前向传播权重文件路径
        :param back_edge_file: 后向传播权重文件路径
        """
        self.learn_alpha = learn_alpha
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.front_edge_file = front_edge_file
        self.back_edge_file = back_edge_file

        # 初始化权重矩阵
        self.front_edge = np.random.randn(self.input_size, self.hidden_size) / np.sqrt(self.hidden_size / 2.0)
        self.back_edge = np.random.randn(self.hidden_size, self.output_size) / np.sqrt(self.output_size / 2.0)

    def save_weights(self, front_edge_file, back_edge_file):
        """
        保存权重矩阵
        :param front_edge_file: 前向权重保存路径
        :param back_edge_file: 反向权重保存路径
        """
        # 保存权重
        np.savetxt(front_edge_file, self.front_edge)
        np.savetxt(back_edge_file, self.back_edge)

    def initialize_weights(self):
        """初始化权重矩阵"""
        self.front_edge = np.random.randn(self.input_size, self.hidden_size) / np.sqrt(self.hidden_size / 2.0)
        self.back_edge = np.random.randn(self.hidden_size, self.output_size) / np.sqrt(self.output_size / 2.0)

    def load_weights(self, front_edge_file, back_edge_file):
        """加载权重矩阵"""
        try:
            self.front_edge = np.loadtxt(front_edge_file)  # 使用传入的文件名
            self.back_edge = np.loadtxt(back_edge_file)    # 使用传入的文件名
        except Exception as e:
            print(f"加载权重失败，错误信息: {e}")
            # 如果加载失败，可以选择重新初始化
            print("开始初始化权重矩阵")
            self.initialize_weights()
            self.save_weights(front_edge_file, back_edge_file)

    def calc_light(self, in_put):
        # 中间节点的计算
        nodes = np.dot(in_put.T, self.front_edge).T
        nodes = np.vectorize(sigmoid)(nodes)  # 非线性化sigmoid(x)

        # 计算最右边亮度
        right_light = np.dot(nodes.T, self.back_edge).T
        right_light = np.vectorize(sigmoid)(right_light)  # 非线性化sigmoid(x)

        return right_light, nodes

    def train_network(self, x_train, y_train, x_val, y_val, front_edge_file, back_edge_file, validation_file):
        """
        训练神经网络
        :param validation_file: 验证结果保存路径
        :param back_edge_file: 反向权重保存路径
        :param front_edge_file: 前向权重保存路径
        :param y_val: 验证集标签
        :param x_val: 验证集数据
        :param x_train: 训练集数据
        :param y_train: 训练集标签
        """
        for i in range(len(x_train)):
            if i % 1000 == 0 and i != 0:
                right_prediction_sum = 0
                confidence = 0
                for j in range(len(x_val)):
                    in_put = np.array(x_val[j])  # 输入数据

                    right_light, nodes = self.calc_light(in_put)

                    predict_value = np.argmax(right_light)
                    if predict_value == y_val[j]:
                        right_prediction_sum += 1
                        confidence += np.max(right_light)
                with open(validation_file, "a", encoding="utf-8") as f:  # 使用 "a" 模式追加写入
                    f.write(f"第{i}次准确率: {right_prediction_sum/len(y_val):.4f}, 置信度: {confidence/right_prediction_sum:.4f}\n")

            # 随机选择一个索引
            print(f"开始训练第{i + 1}组数据")

            # 训练数据 in_put
            in_put = np.array(x_train[i])  # 输入数据

            # 前向传播计算节点值
            right_light, nodes = self.calc_light(in_put)

            # 定义目标亮度
            tar_nodes = np.zeros((1, self.output_size))  # 定义目标亮度

            # 给目标亮度赋值（标签集）
            for j in range(self.output_size):
                if j == y_train[i]:
                    tar_nodes[0][j] = 0.999  # 目标类别设置为 0.999
                else:
                    tar_nodes[0][j] = 0.001  # 非目标类别设置为 0.001

            # 反向传播更新后边权重
            self.back_edge, dif_value = calc_weight(nodes, right_light, tar_nodes, self.hidden_size, self.output_size,
                                                    self.back_edge, self.learn_alpha)

            # 反向传播更新节点值
            tar_nodes = np.dot(self.back_edge, dif_value)

            # 反向传播更新前边权重
            self.front_edge, dif_value = calc_weight(in_put, nodes, tar_nodes.T, self.input_size, self.hidden_size,
                                                     self.front_edge, self.learn_alpha)

            # 保存权重文件
            self.save_weights(front_edge_file, back_edge_file)

    def test_network(self, x_test, y_test, front_edge_file, back_edge_file, result_file):
        """
        测试神经网络
        :param x_test: 测试集输入
        :param y_test: 测试集标签
        :param front_edge_file: 前向权重保存路径
        :param back_edge_file: 反向权重保存路径
        :param result_file: 测试结果保存路径
        """
        self.load_weights(front_edge_file, back_edge_file)

        right_prediction_sum = 0
        confidence = 0

        for i in range(len(y_test)):
            print(f"开始测试第{i}组数据")

            in_put = np.array(x_test[i])  # 输入数据

            right_light, nodes = self.calc_light(in_put)

            predict_value = np.argmax(right_light)
            if predict_value == y_test[i]:
                right_prediction_sum += 1
                confidence += np.max(right_light)

        with open(result_file, "w", encoding="utf-8") as f:  # 使用 "a" 模式追加写入
            f.write(f"准确率: {right_prediction_sum / len(y_test):.4f}, 置信度: {confidence / right_prediction_sum:.4f}\n")

def run_experiment(learn_alpha, hidden_size, input_size=784, output_size=10,
                   front_edge_file="FrontEdge.txt", back_edge_file="BackEdge.txt",
                   result_file="result.txt", validation_file="Validation.txt"):
    """
    运行一次完整的实验（训练 + 测试）
    :param learn_alpha: 学习率
    :param hidden_size: 隐藏层大小
    :param input_size: 输入层大小（默认 784）
    :param output_size: 输出层大小（默认 10）
    :param front_edge_file: 前向权重保存路径
    :param back_edge_file: 反向权重保存路径
    :param result_file: 测试结果保存路径
    :param validation_file: 验证结果保存路径
    """
    # 加载数据集
    new_data = GetData().create_dataset()

    # 初始化网络
    bp_network = BpNetwork(learn_alpha, hidden_size, input_size, output_size, front_edge_file, back_edge_file)

    # 加载权重（如果文件存在）
    bp_network.load_weights(front_edge_file, back_edge_file)

    # 训练网络
    bp_network.train_network(
        new_data.x_train, new_data.y_train,
        new_data.x_val, new_data.y_val,
        front_edge_file, back_edge_file,
        validation_file
    )

    # 测试网络
    bp_network.test_network(
        new_data.x_test, new_data.y_test,
        front_edge_file, back_edge_file,
        result_file
    )

    # 保存验证结果
    with open(validation_file, "a", encoding="utf-8") as f:
        f.write(f"实验配置: 学习率={learn_alpha}, 隐藏层大小={hidden_size}\n")

# 运行实验
run_experiment(
    learn_alpha=1,
    hidden_size=100,
    front_edge_file="1-100_FrontEdge.txt",
    back_edge_file="1-100_BackEdge.txt",
    result_file="1-100_result.txt",
    validation_file="1-100_Validation.txt"
)

run_experiment(
    learn_alpha=1,
    hidden_size=200,
    front_edge_file="1-200_FrontEdge.txt",
    back_edge_file="1-200_BackEdge.txt",
    result_file="1-200_result.txt",
    validation_file="1-200_Validation.txt"
)

run_experiment(
    learn_alpha=2,
    hidden_size=100,
    front_edge_file="2-100_FrontEdge.txt",
    back_edge_file="2-100_BackEdge.txt",
    result_file="2-100_result.txt",
    validation_file="2-100_Validation.txt"
)

run_experiment(
    learn_alpha=2,
    hidden_size=200,
    front_edge_file="2-200_FrontEdge.txt",
    back_edge_file="2-200_BackEdge.txt",
    result_file="2-200_result.txt",
    validation_file="2-200_Validation.txt"
)


