import zipfile
import os

# 压缩文件路径
zip_path='data.zip'

# 文件存储路径
save_path = '.'

# 读取压缩文件
file=zipfile.ZipFile(zip_path)

# 解压文件
print('开始解压...')
file.extractall(save_path)
print('解压结束。')