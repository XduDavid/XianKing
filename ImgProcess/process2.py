import os
import random
from PIL import Image
from numpy import ndarray
# image processing library
import skimage as sk
from skimage import transform
from skimage import util
from skimage import io
from skimage.filters import unsharp_mask
from skimage.filters import gaussian
from skimage.transform import resize
import matplotlib.pyplot as plt

def random_rotation(image_array: ndarray):
    # 旋转幅度+25%到-25%
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)

def random_noise(image_array: ndarray):
    # 随机噪音
    return sk.util.random_noise(image_array)

def random_sharpen(image_array: ndarray):
    # 随机锐化
    random_radius = random.uniform(1, 20)
    random_amount = random.uniform(1, 5)
    return unsharp_mask(image_array, radius=random_radius, amount=random_amount)

def random_gasussian(image_array: ndarray):
    # 随机模糊
    random_sigma = random.uniform(0, 5)
    return gaussian(image_array, sigma=random_sigma, multichannel=True)

# dictionary of the transformations we defined earlier
available_transformations = {
    'rotate': random_rotation,
    'noise': random_noise,
    'sharpen': random_sharpen,
    'gaussian': random_gasussian
}

folder_path = '.\\2'
num_files_desired = 200 #生成数量

# 储存文件路径
images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

num_generated_files = 0
while num_generated_files <= num_files_desired:
    # 随机选择图片
    image_path = random.choice(images)
    # 以二维矩阵读入
    image_to_transform = sk.io.imread(image_path)
    # 随机效果编号选择
    num_transformations_to_apply = random.randint(1, len(available_transformations))

    num_transformations = 0
    transformed_image = None
    while num_transformations <= num_transformations_to_apply:
        # 随机效果选择
        key = random.choice(list(available_transformations))
        transformed_image = available_transformations[key](image_to_transform)
        num_transformations += 1

        new_file_path = '%s/augmented_image_%s.png' % (folder_path, num_generated_files)
        # 写回加重新拉伸为128*128的png照片
        io.imsave(new_file_path, resize(transformed_image, (128, 128), anti_aliasing=True))
    num_generated_files += 1