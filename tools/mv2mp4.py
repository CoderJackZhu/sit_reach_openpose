import os
import shutil

source_dir = 'E:/Project/Sit_and_reach/data'
save_path = 'E:/Project/Sit_and_reach_video'

if not os.path.exists(save_path):
    os.mkdir(save_path)
for file in os.listdir(source_dir):
    video = os.path.join(source_dir, file, file[-3:] + '.mkv')
    # save_video=os.path.join(save_path, file+'.mkv')
    shutil.copy(video, save_path)
    print(video)
