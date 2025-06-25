import torch
import os 
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
# 检查 CUDA 是否可用
if torch.cuda.is_available():
    # 获取 CUDA 设备数量
    device_count = torch.cuda.device_count()
    print("CUDA 设备数量：", device_count)

    # 遍历每个 CUDA 设备并打印设备编号
    for i in range(device_count):
        device = torch.device("cuda", i)
        print("CUDA 设备编号：", i, "设备名称：", torch.cuda.get_device_name(i))
else:
    print("CUDA 不可用")
