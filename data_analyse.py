import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

file = pd.read_csv('./result.csv', header=0)
# data = file.iloc[:, 0:3]
data = file.iloc[:, 0:4]
none_len = np.sum(data.iloc[:, 1] == -1)
print(f'未找到手的个数为{none_len}')
absolute_error_list = []
relative_error_list = []
for i in range(data.shape[0]):
    if data.iloc[i, 3] != -1:
        absolute_error = data.iloc[i, 3] - data.iloc[i, 2]
        if np.abs(absolute_error) > 10:
            print(i, absolute_error)
        relative_error = absolute_error / data.iloc[i, 2]
        absolute_error_list.append(absolute_error)
        relative_error_list.append(relative_error)
data.iloc[:, 3].hist()
plt.title('data histogram')
plt.show()
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
