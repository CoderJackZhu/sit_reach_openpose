import copy
import os

import cv2
import numpy as np
import pandas as pd

from src import util
from src.body import Body
from src.hand import Hand

from librs.one_detect import plot_one
from librs.utils import get_x_dis

ins = np.array([
    [[745, 473], [1176, 478]],
    [[785, 450], [1212, 446]],
    [[772, 445], [1183, 443]],
    [[805, 449], [1251, 451]]
])

bg = np.array([
    [[326, 247], [1245, 708]],
    [[277, 160], [1273, 709]],
    [[228, 129], [1253, 700]],
    [[310, 188], [1276, 716]]
])

keen_y = [470, 450, 445, 450]

def cal_multi_vid(root_dir='./output', save_path='./test'):
    ori_data = pd.read_csv('source_data.csv', header=0)
    files = os.listdir(root_dir)
    files.sort()
    row_list = ['label_frame', 'meas_frame', 'label_len', 'meas_len']
    result_file = pd.DataFrame(columns=row_list)
    for file in files:
        id = int(file[-3:])
        frame = ori_data.iloc[id - 1, 3]
        mode = ori_data.iloc[id - 1, 5] - 1
        pics = os.listdir(os.path.join(root_dir, file))
        pics.sort()
        hand_loc_list = []
        for pic in pics:
            picture = os.path.join(os.path.join(root_dir, file, pic))
            if not os.path.exists(os.path.join(save_path, str(id))):
                os.makedirs(os.path.join(save_path, str(id)))
            hand = plot_one(picture, save_file=os.path.join(save_path, str(id), f'{pic}'), keen_y=keen_y[mode])
            if len(hand) != 0:
                hand_location = hand[0][12]
            else:
                hand_location = [0, 0]
            hand_loc_list.append(hand_location)
            print(f'file {file}:{pic[:-4]}/{len(pics)} finished ')
        hand_loc_list = np.array(hand_loc_list)
        large_site = np.argmax(hand_loc_list, axis=0)[0]
        large_hand_x, large_hand_y = hand_loc_list[large_site, 0], hand_loc_list[large_site, 1]
        if ins[mode, 0, 0] < large_hand_x < ins[mode, 1, 0] and \
                bg[mode, 0, 1] < large_hand_y < bg[mode, 1, 1]:
            print('最远手的坐标是({},{})'.format(large_hand_x, large_hand_y))
            result = (large_hand_x - ins[mode, 0, 0]) / (ins[mode, 1, 0] - ins[mode, 0, 0]) * 80
            result = result.round(2)
            print('记录成绩为{}'.format(ori_data.iloc[id - 1, 4]))
            print(f'预测成绩为{result}')
            print(f'最远帧为{large_site + 1}')
        else:
            result = 0
            print('最远手的位置不在范围内')
        result_file.loc[id] = [frame, large_site + 1, ori_data.iloc[id - 1, 4], result]
        result_file.to_csv('./show_result/result_file.csv', index=False)
    # with open('./result.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(row_list)


if __name__ == '__main__':
    cal_multi_vid()
