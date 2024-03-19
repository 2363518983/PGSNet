import os

# 检查 NVIDIA 驱动程序是否已安装
if not os.path.exists('/usr/bin/nvidia-smi'):
    print('NVIDIA 驱动程序未安装')
else:
    # 检查 NVIDIA 驱动程序是否正在运行
    try:
        os.system('nvidia-smi')
    except:
        print('NVIDIA 驱动程序未正确运行')