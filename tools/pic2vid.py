import os
import sys
import subprocess


def process_pic(root_dir='../output/20220628014', save_dir='../test', fps=30):
    save_file = os.path.join(save_dir, 'output.mp4')
    cmd = 'ffmpeg -f image2 -r {} -i {}/%03d.jpg -b:v 4M {}  -crf 1'.format(fps, root_dir, save_file)
    subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == '__main__':
    process_pic(root_dir='../result2/20220628029', save_dir='../result2', fps=30)
