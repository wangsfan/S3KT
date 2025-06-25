import imageio
import os

image_folder = './result/b/'   # 设定要合成的图像所在的文件夹路径
images = []
for file_name in os.listdir(image_folder):
    if file_name.endswith('.jpg'):
        file_path = os.path.join(image_folder, file_name)
        images.append(imageio.imread(file_path))
imageio.mimsave(os.path.join(image_folder, 'movie.gif'), images, fps=100)  # 修改fps可以控制gif的播放速度