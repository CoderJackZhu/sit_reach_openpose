import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

file = pd.read_csv('./坐位体前屈算法结果.csv', header=None)
data = file.iloc[:, 0:3]
none_len = np.sum(data.iloc[:, 1] == -1)
print(f'未找到手的个数为{none_len}')
absolute_error_list = []
relative_error_list = []
for i in range(data.shape[0]):
    if data.iloc[i, 1] != -1:
        absolute_error = data.iloc[i, 1] - data.iloc[i, 0]
        relative_error = absolute_error / data.iloc[i, 0]
        absolute_error_list.append(absolute_error)
        relative_error_list.append(relative_error)
# data.iloc[:,1].hist()
average_error = sum(map(abs, absolute_error_list)) / len(absolute_error_list)
print('平均误差为{:.2f}'.format(average_error))
plt.hist(absolute_error_list)
plt.title('absolute_error')
plt.show()
plt.hist(relative_error_list)
plt.title('relative_error')
plt.show()
fig, axes = plt.subplots()
sns.boxplot(data=absolute_error_list, orient='v', ax=axes)
plt.show()
