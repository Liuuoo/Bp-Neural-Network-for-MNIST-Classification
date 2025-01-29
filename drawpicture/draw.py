import matplotlib.pyplot as plt
import re
import os
from matplotlib.font_manager import FontProperties  # 确保中文字体支持


def draw_multiple_plots(txt_folder, target_file, labels=None, title="多组数据对比图"):
    """
    自动读取指定文件夹内的所有txt文件，并绘制准确率和置信度对比图

    参数:
        txt_folder (str): 存放txt文件的文件夹路径（例如 "./txt"）
        target_file (str): 输出图片的保存路径（例如 "combined_plot.png"）
        labels (list): 自定义数据标签（可选，默认使用文件名）
        title (str): 图表标题（可选）
    """
    # 设置中文字体（根据系统调整）
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows
    plt.rcParams['axes.unicode_minus'] = False

    # 获取文件夹内所有txt文件
    if not os.path.exists(txt_folder):
        raise FileNotFoundError(f"文件夹 {txt_folder} 不存在")
    txt_files = [f for f in os.listdir(txt_folder) if f.endswith(".txt")]
    if not txt_files:
        raise ValueError(f"文件夹 {txt_folder} 中没有txt文件")
    txt_files = sorted(txt_files)  # 按文件名排序

    # 生成默认标签（去除文件后缀）
    if labels is None:
        labels = [os.path.splitext(f)[0] for f in txt_files]
    elif len(labels) != len(txt_files):
        raise ValueError("labels参数长度与文件数量不一致")

    # 初始化画布和样式
    plt.figure(figsize=(14, 6))
    colors = ['skyblue', 'lightcoral', 'mediumseagreen', 'orchid', 'goldenrod']  # 浅色系
    line_styles = ['-', '--', '-.', ':']  # 线型区分

    # 循环处理每个文件
    for idx, (filename, label) in enumerate(zip(txt_files, labels)):
        filepath = os.path.join(txt_folder, filename)
        iterations, accuracy, confidence = [], [], []

        # 读取数据
        with open(filepath, "r", encoding="utf-8") as file:
            for line in file:
                matches = re.findall(r"\d+\.?\d*", line)
                if len(matches) == 3:
                    iterations.append(int(matches[0]))
                    accuracy.append(float(matches[1]))
                    confidence.append(float(matches[2]))

        # 绘制到子图
        color = colors[idx % len(colors)]
        linestyle = line_styles[idx % len(line_styles)]

        # 准确率子图
        plt.subplot(1, 2, 1)
        plt.plot(
            iterations, accuracy,
            color=color,
            linestyle=linestyle,
            linewidth=1.2,
            alpha=0.8,
            label=label
        )

        # 置信度子图
        plt.subplot(1, 2, 2)
        plt.plot(
            iterations, confidence,
            color=color,
            linestyle=linestyle,
            linewidth=1.2,
            alpha=0.8,
            label=label
        )

    # 统一设置子图属性
    for i in [1, 2]:
        plt.subplot(1, 2, i)
        plt.xlabel("迭代轮数", fontsize=10)
        plt.ylabel("准确率" if i == 1 else "置信度", fontsize=10)
        plt.grid(True, linestyle=":", alpha=0.4)
        plt.legend(fontsize=8, loc="upper left", frameon=False)  # 简洁图例

    # 全局标题和保存
    plt.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(target_file, dpi=300, bbox_inches="tight")
    plt.close()


# 示例调用
if __name__ == "__main__":
    draw_multiple_plots(
        txt_folder="./txt",  # 根目录下的txt文件夹
        target_file="对比图.png",  # 输出图片路径
        title="训练指标对比（4组数据）"
    )