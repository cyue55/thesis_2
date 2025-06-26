import numpy as np

def inspect_scores_file():
    """Inspect the content of a scores.npz file"""
    # 读取一个示例文件
    path = "/Users/yue/Documents/Thesis/Codes/thesis/recipes/cnrhlc/models/bsrnn-nr-l1-nalr/scores.npz"
    data = np.load(path)
    
    # 打印文件中的所有键和对应数据的形状
    print("Keys in scores.npz:")
    for key in data.keys():
        print(f"Key: {key}")
        print(f"Shape: {data[key].shape}")
        print(f"Sample values: {data[key][:3]}\n")

if __name__ == "__main__":
    inspect_scores_file()
