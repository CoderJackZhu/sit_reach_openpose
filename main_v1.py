import copy
import csv
import os

import cv2
import numpy as np
import pandas as pd

from src import util
from src.body import Body
from src.hand import Hand

from librs.utils import plot_one, get_x_dis, cal_one_best_result, get_TAL_frame
from librs.configs import *


def cal_multi_vid(choose='predict', root_dir='E:/Project/Sit_and_reach_clip', save_path='./test2'):
    ori_data = pd.read_csv('source_data.csv', header=0)
    files = os.listdir(root_dir)
    files.sort()
    row_list = ['label_frame', 'meas_frame', 'label_len', 'meas_len', 'x_y']
    result_file = pd.DataFrame(columns=row_list)
    for file in files:
        id = int(file[-3:])
        data = ori_data.iloc[id - 1]
        frame = int(data[3])
        mode = int(data[5] - 1)
        if type(choose) == str and choose == 'all':
            pics = os.listdir(os.path.join(root_dir, file))
        elif type(choose) == int:
            pics = [str(frame).zfill(3) + '.jpg' for frame in range(frame - choose // 2, frame + choose // 2 + 1)]
        elif choose == 'predict':
            predict_frame = get_TAL_frame(id - 1)
            if 0 < predict_frame < data[7]:
                pics = [str(int(predict_frame)).zfill(3) + '.jpg']
            else:
                result_file.loc[id] = [-2, -2, -2, -2, [-2, -2]]
                continue
        else:
            raise ValueError('choose must be int or str')
        pics.sort()
        canvas, far_frame, x_y, score = cal_one_best_result(data=data, root_dir=root_dir,
                                                            file=file, pics=pics, mode=mode, save_path=save_path)
        result_file.loc[id] = [frame, far_frame, data[4], score, x_y]
        result_file.to_csv('./show_result/predict_result.csv', index=False)
    with open('./show_result/all_predict_result.csv', 'w', newline='') as f:
        f.write(result_file.to_csv(index=False))


if __name__ == '__main__':
    cal_multi_vid(choose='predict', root_dir='./output', save_path='./desti_test')
