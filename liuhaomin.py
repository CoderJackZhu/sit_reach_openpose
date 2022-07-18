import cv2
import numpy as np
import pandas as pd

src = cv2.imread('E:\\Project\\try\\tiaoyuan\\transformation\\143.jpg')
w, h = 1280, 500
srcPoint = np.float32([[420, 449], [770, 449], [133, 719], [933, 719]])  # 场景2，参考线内缘四角
canPoint = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

Mat = cv2.getPerspectiveTransform(np.array(srcPoint), np.array(canPoint))

result = cv2.warpPerspective(src, Mat, (1280, 500))


# src6 = cv2.resize(src6,(300,400))

# cv2.imshow('src',src)
# cv2.imshow('res',result)
# cv2.imwrite('142_110_r.jpg',result)
# cv2.waitKey(0)
# 坐标变换
def cvt_pos(pos, cvt_mat_t):
    u = pos[0]
    v = pos[1]
    x = (cvt_mat_t[0][0] * u + cvt_mat_t[0][1] * v + cvt_mat_t[0][2]) / (
            cvt_mat_t[2][0] * u + cvt_mat_t[2][1] * v + cvt_mat_t[2][2])
    y = (cvt_mat_t[1][0] * u + cvt_mat_t[1][1] * v + cvt_mat_t[1][2]) / (
            cvt_mat_t[2][0] * u + cvt_mat_t[2][1] * v + cvt_mat_t[2][2])
    return (x, y)


data = pd.read_csv('30.csv')
ori_data = data.values.tolist()
list1 = []
list2 = []
result = []
accuracy = []
gap = []
for i in range(141, 157):
    list1 = ori_data[i][1]
    list2 = ori_data[i][2]
    sorce = ori_data[i][4]
    list1 = list1[1:-1]
    list2 = list2[1:-1]
    pos_l = []
    pos_r = []
    for j in range(2):
        a = list1.split(',')[j]
        b = list2.split(',')[j]
        a = float(a)
        b = float(b)
        pos_l.append(a)
        pos_r.append(b)
    x_l, y_l = cvt_pos(pos_l, Mat)
    x_r, y_r = cvt_pos(pos_r, Mat)
    # print('result',i+1)
    # print('左脚：',(x_l,y_l))
    # print('右脚：',(x_r,y_r))
    # if (pos_l[0] >= pos_r[0])|(pos_l[0] == 0) :
    if x_l >= x_r:
        x = x_r
        y = y_r
    # elif (pos_l[0] < pos_r[0])|(pos_r[0] == 0):
    else:
        x = x_l
        y = y_l
    print("落地点：", (x, y))

    dis = 200  # 参考线之间的距离
    r = x * dis / w
    print('成绩：', r)
    result.append(r)
    c = abs(r - sorce)
    acc = c / sorce
    print(acc)
    accuracy.append(acc)
    gap.append(c)
