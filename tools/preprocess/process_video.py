from rembg.bg import remove
import numpy as np
import io
from PIL import Image
import configargparse
import os
import cv2
import mmcv
from PIL import ImageFile
import argparse


ImageFile.LOAD_TRUNCATED_IMAGES = True

def add_white_bg(input_path, masks_out_path, white_bg=False):
    if not os.path.exists(masks_out_path):
        os.makedirs(masks_out_path)
    for name in os.listdir(input_path):
        dir = os.path.join(input_path, name)
        im = Image.open(dir)
        if white_bg:
            mask = (np.array(im).mean(-1) != 255) * 255
        else:
            mask = (np.array(im)[:,:,-1] > 128) * 255
        cv2.imwrite(os.path.join(masks_out_path, name), mask.astype(np.uint8))
    print("Done with masks saved at {}.".format(masks_out_path))


def get_frames(filename='test.mp4', output_path='./', interval=10):

    print("Spliting video to frames with an interval of {} ...".format(interval))
    video = mmcv.VideoReader(filename)

    # obtain basic information
    print(len(video))
    print(video.width, video.height, video.resolution, video.fps)

    img = video[0:-1:interval]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for i in range(len(img)):
        name = os.path.join(output_path, '%05d'%i + '.jpg')
        cv2.imwrite(name, img[i])
    print("Done with {} frames.".format(len(img)))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='get_frames')
    parser.add_argument('--source_dir', type=str, help='data source folder for preprocess')
    parser.add_argument('--video_path', type=str, help='video to process')
    parser.add_argument('--img_folder', type=str, default='image')
    parser.add_argument('--rmbg_img_folder', type=str, default='image_rmbg')
    parser.add_argument('--interval', type=int, default=10)
    parser.add_argument('--white_bg', action='store_true')
    opt = parser.parse_args()

    root = opt.source_dir
    images_ori_path = os.path.join(root, opt.img_folder)
    images_out_path = os.path.join(root, opt.rmbg_img_folder)
    masks_out_path = os.path.join(root, 'mask')

    if opt.mode == 'get_frames':
        get_frames(opt.video_path, images_ori_path, interval=opt.interval)
    elif opt.mode == 'get_masks':
        # remove_bg(images_ori_path, images_out_path, masks_out_path)
        add_white_bg(images_out_path, masks_out_path, opt.white_bg)
    else:
        raise NameError
