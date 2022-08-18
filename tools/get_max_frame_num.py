import pandas as np
import numpy as np
import cv2
import os
import csv


def get_max_frame(root_dir='E:/Project/Sit_and_reach_clip'):
    files = os.listdir(root_dir)
    files.sort()
    frame_list = []
    for file in files:
        pics = os.listdir(os.path.join(root_dir, file))
        pics.sort()
        frame = int(pics[-1][:-4])
        print(frame)
        frame_list.append(frame)
    return frame_list


def frames_to_csv(frame_list):
    with open('../show_result/frames_nums.csv', 'w', newline='', ) as f:
        writer = csv.writer(f)
        for i in range(len(frame_list)):
            writer.writerow([frame_list[i]])


if __name__ == '__main__':
    frames_list = get_max_frame()
    frames_to_csv(frames_list)
