import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calc_weight(left_light, right_light, ans, row, lie, edge_weight, learn_alpha):
    """
    计算并更新权重
    :param left_light: 前一层节点的输出值，形状为 (row, 1)
    :param right_light: 当前层节点的输出值，形状为 (lie, 1)
    :param ans: 目标值，形状为 (1, lie)
    :param row: 前一层节点数
    :param lie: 当前层节点数
    :param edge_weight: 权重矩阵，形状为 (row, lie)
    :param learn_alpha: 学习率
    :return: 更新后的权重矩阵和误差值
    """
    # 计算误差值，dif_value = y_true - y_pred
    dif_value = (ans.T - right_light)  # 形状为 (lie, 1)

    # 计算梯度
    gradient = right_light * (1 - right_light) * dif_value  # 形状为 (lie, 1)

    # 更新权重
    edge_weight += learn_alpha * np.dot(left_light, gradient.T)  # 形状为 (row, lie)

    return edge_weight, dif_value