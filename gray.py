# import h5py
# import numpy as np
# from PIL import Image

# hdf5path = '/media/thc/Elements/outdoor_day/outdoor_day1_data.hdf5'
# data = h5py.File(hdf5path, 'r')
# # data1 = data['davis']['left'][]
# gray_gt = np.array(data['davis']['left']['image_raw'])
# for i in range(len(gray_gt)):
    
#     image = Image.fromarray(gray_gt[i], 'L')

#     # 保存图像为JPG格式
#     image.save('/media/thc/Elements/outdoor_day/outdoor_day1_gray/'+ str(i) + '.jpg')
# # print(data1.keys())
from PIL import Image

# 打开图像文件
image = Image.open('/media/thc/Elements/dsec/train_images/zurich_city_01_a/images/left/rectified/000001.png')

# 将图像转换为灰度图像
gray_image = image.convert('L')


# 保存灰度图像
gray_image.save('gray_image.jpg')
