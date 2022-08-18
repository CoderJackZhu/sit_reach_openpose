import copy
import os
import cv2
import numpy as np
from tqdm import tqdm
from librs.one_detect import plot_one


def plot_multi(output_dir='./test2', result_dir='./result2'):
    files = os.listdir(output_dir)
    for file in tqdm(files):
        if not os.path.exists(os.path.join(result_dir, file)):
            os.mkdir(os.path.join(result_dir, file))
        pics = os.listdir(os.path.join(output_dir, file))
        for pic in tqdm(pics):
            picture = os.path.join(os.path.join(output_dir, file, pic))
            print(picture)
            # plot_one(test_image=picture, save_file=os.path.join(result_dir, file, pic), keen_y=470)
            plot_one(test_image=picture, save_file=os.path.join(result_dir, file, pic))
