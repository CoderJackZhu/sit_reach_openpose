import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def analyse_data(data_path='../show_result/predict_interframe_result.csv'):
    file = pd.read_csv(data_path, header=0)
    # data = file.iloc[:, 0:3]
    data = file.iloc[:, 0:4]
    none_len = np.sum(data.iloc[:, 1] == -1)
    print(f'未找到手的个数为{none_len}')
    absolute_error_list = []
    relative_error_list = []
    frame_abs_error_list, frame_rel_error_list = [], []
    for i in range(data.shape[0]):
        # if data.iloc[i, 3] != -1:
        absolute_error = data.iloc[i, 3] - data.iloc[i, 2]
        if np.abs(absolute_error) > 10:
            print(i, absolute_error)
        relative_error = absolute_error / data.iloc[i, 2]
        absolute_error_list.append(absolute_error)
        relative_error_list.append(relative_error)
        frame_abs_error = data.iloc[i, 1] - data.iloc[i, 0]
        frame_rel_error = frame_abs_error / data.iloc[i, 0]
        frame_abs_error_list.append(frame_abs_error)
        frame_rel_error_list.append(frame_rel_error)
    data.iloc[:, 3].hist()
    plt.title('data histogram')
    plt.show()
    average_error = sum(map(abs, absolute_error_list)) / len(absolute_error_list)
    print('测量结果平均误差为{:.2f}'.format(average_error))
    plt.hist(absolute_error_list)
    plt.title('absolute_error')
    plt.show()
    plt.hist(relative_error_list)
    plt.title('relative_error')
    plt.show()
    fig, axes = plt.subplots()
    sns.boxplot(data=absolute_error_list, orient='v', ax=axes)
    plt.title('absolute_error boxplot')
    plt.show()

    frame_average_error = sum(map(abs, frame_abs_error_list)) / len(frame_abs_error_list)
    print('预测帧平均误差为{:.2f}'.format(frame_average_error))
    plt.hist(frame_abs_error_list)
    plt.title('frame_absolute_error')
    plt.show()
    plt.hist(frame_rel_error_list)
    plt.title('frame_relative_error')
    plt.show()
    fig1, axes = plt.subplots()
    sns.boxplot(data=frame_abs_error_list, orient='v', ax=axes)
    plt.title('frame_absolute_error boxplot')
    plt.show()


def analyse_frame(data_path='../show_result/predict_result_v1.csv'):
    file = pd.read_csv(data_path, header=0)
    with open('../show_result/sit_result_TAL.txt', 'r') as f:
        frame_data = f.readlines()
    frame_data = [i[:-1].split(' ') for i in frame_data]
    frame_abs_error_list, frame_rel_error_list = [], []
    for i in range(file.shape[0]):
        frame_abs_error = float(frame_data[i][1]) - file.iloc[i, 0]
        frame_rel_error = frame_abs_error / file.iloc[i, 0]
        frame_abs_error_list.append(frame_abs_error)
        frame_rel_error_list.append(frame_rel_error)

    frame_average_error = sum(map(abs, frame_abs_error_list)) / len(frame_abs_error_list)
    print('预测帧平均误差为{:.2f}'.format(frame_average_error))
    plt.hist(frame_abs_error_list)
    plt.title('frame_absolute_error')
    plt.show()
    plt.hist(frame_rel_error_list)
    plt.title('frame_relative_error')
    plt.show()
    fig1, axes = plt.subplots()
    sns.boxplot(data=frame_abs_error_list, orient='v', ax=axes)
    plt.title('frame_absolute_error boxplot')
    plt.show()


if __name__ == '__main__':
    analyse_data()
