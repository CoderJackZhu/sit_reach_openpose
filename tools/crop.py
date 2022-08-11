import os

import cv2

if __name__ == '__main__':
    img_dir = 'E:/Project/Sit_and_reach_clip'
    cropped_dir = '../cropped'
    img_list = os.listdir(img_dir)
    for img in img_list:
        img_path = os.path.join(img_dir, img)
        save_path = os.path.join(cropped_dir, img)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        files = os.listdir(img_path)
        for file in files:
            file_path = os.path.join(img_path, file)
            pic = cv2.imread(file_path, 1)
            dst = pic[247:708, 326:1245]  # 裁剪坐标为[y0:y1, x0:x1]
            cv2.imshow('image', dst)
            # cv2.waitKey(0)
            cv2.imwrite(os.path.join(save_path, file), dst)
        print('{} is done'.format(img))
