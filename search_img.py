from PIL import Image
import numpy as np
import os
import shutil

def compute_image_similarity(image_path1, image_path2):
    """
    计算两张图片的相似度
    """
    img1 = Image.open(image_path1)
    img2 = Image.open(image_path2)
    h1 = img1.histogram()
    h2 = img2.histogram()
    rms = np.sqrt(np.mean(np.subtract(h1, h2)**2))
    return rms

def find_similar_images(dataset_dir, data_dir, threshold=10):
    """
    从数据集中查找与样本图片相似度大于指定阈值的图片，并将其保存到 out 目录中
    """
    for filename1 in os.listdir(data_dir):
        image_path1 = os.path.join(data_dir, filename1)
        for filename2 in os.listdir(dataset_dir):
            image_path2 = os.path.join(dataset_dir, filename2)
            similarity = compute_image_similarity(image_path1, image_path2)
            if similarity < threshold:
                out_path = os.path.join("out", filename2)
                shutil.copy(image_path2, out_path)

if __name__ == "__main__":
    dataset_dir = "dataset"
    data_dir = "data"
    out_dir = "out"
    threshold = 10
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    find_similar_images(dataset_dir, data_dir, threshold)