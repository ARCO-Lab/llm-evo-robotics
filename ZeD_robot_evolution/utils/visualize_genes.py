import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET
import os

def load_best_genes():
    """加载最新的 best_robot_gen_XXX.xml"""
    files = sorted([f for f in os.listdir() if f.startswith("best_robot_gen_")], key=lambda x: int(x.split("_")[-1].split(".")[0]))
    if not files:
        print("❌ 没有找到最优基因文件！")
        return None
    latest_file = files[-1]
    tree = ET.parse(latest_file)
    root = tree.getroot()

    genes = [int(body.get("name").split("_")[-1]) for body in root.findall(".//body")]
    return np.array(genes)

def visualize_genes():
    genes = load_best_genes()
    if genes is None:
        return
    plt.bar(range(len(genes)), genes)
    plt.xlabel("Component ID")
    plt.ylabel("Selected (1) or Not (0)")
    plt.title("Best Robot Genetic Encoding")
    plt.show()

visualize_genes()


if __name__ == "__main__":
    print("✅ 测试基因可视化...")
    visualize_genes()

