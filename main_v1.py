import copy
import os

import cv2
import numpy as np
import pandas as pd

from src import util
from src.body import Body
from src.hand import Hand

from librs.utils import plot_one, cal_one_best_result
from librs.utils import get_x_dis
from librs.configs import *


def cal_multi_vid(root_dir='./output', save_path='./test'):
    ori_data = pd.read_csv('source_data.csv', header=0)
    files = os.listdir(root_dir)
    files.sort()
    row_list = ['label_frame', 'meas_frame', 'label_len', 'meas_len', 'x_y']
    result_file = pd.DataFrame(columns=row_list)
    for file in files:
        id = int(file[-3:])
        data = ori_data.iloc[id - 1]
        frame = data[3]
        mode = int(data[5] - 1)
        pics = os.listdir(os.path.join(root_dir, file))
        pics.sort()
        far_frame, x_y, score = cal_one_best_result(data=data, root_dir=root_dir,
                                                    file=file, pics=pics, mode=mode, save_path=save_path)
        result_file.loc[id] = [frame, far_frame, data[4], score, x_y]
        result_file.to_csv('./show_result/all_data_result.csv', index=False)
    # with open('./result.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(row_list)


if __name__ == '__main__':
    cal_multi_vid()
