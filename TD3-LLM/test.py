import numpy as np
# import numpy as np

# # 从 txt 文件中读取数据并恢复为原来的数组列表
# def read_arrays_from_txt(filename):
#     arrays = []
#     with open(filename, 'r') as f:
#         arrays = f.read()
#     return arrays

# # 读取数据
# loaded_list = read_arrays_from_txt('data.txt')

# # 打印恢复后的列表
# print(loaded_list)
# print(type(loaded_list))

gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / 20]]
for m in range(20 - 1):
    gaps.append(
        [gaps[m][1], gaps[m][1] + np.pi / 20]
    )
gaps[-1][-1] += 0.03
print(gaps)