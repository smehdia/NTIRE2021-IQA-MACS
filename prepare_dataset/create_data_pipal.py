import os
import cv2
import numpy as np
from tqdm import tqdm

'''
This code converts all images in the PIPAL dataset into 
 two Numpy files (the training images and the scores)
 The format of the saved arrays is like the format of the saved H5 files for PieAPP and TID datasets, 
 to understand the format please refer to create_data_pieapp_tid.py comments
'''

PIPAL_DIR = "../../../data"
IMG_SIZE = 288
NUM_CHANNELS = 3
NUMPY_SAVED_IMAGES = 'total_inputs'
NUMPY_SAVED_SCORES = 'total_scores'

def get_num_imgs(path):
    i = 0
    for root, dirs, files in os.walk(path):
        for img_name in files:
            if img_name.split('.')[-1] != 'bmp':
                continue
            i = i + 1

    return i


if __name__ == "__main__":

    path = PIPAL_DIR + '/training_inp'
    labels_path = PIPAL_DIR + '/Train_Label'
    ref_path = PIPAL_DIR + '/Train_Ref'
    num_imgs = get_num_imgs(path)

    x = np.zeros(shape=(num_imgs, 2, IMG_SIZE, IMG_SIZE, NUM_CHANNELS), dtype='uint8')
    y = np.zeros(shape=(num_imgs))

    i = 0
    for root, dirs, files in os.walk(path):
        for img_name in tqdm(files):
            if img_name.split('.')[-1] != 'bmp':
                continue
            img_dir = os.path.join(root, img_name)

            img_ref = img_name.split('_')[0]
            ref_file = open(labels_path + '/' + img_ref + '.txt', 'r')
            score = float(list(filter(lambda x: img_name in x, ref_file.readlines()))[0].split(',')[1])

            distorted_img = cv2.imread(img_dir)
            ref_img = cv2.imread(ref_path + '/' + img_ref + '.bmp')

            x[i, 0] = distorted_img
            x[i, 1] = ref_img
            y[i] = score
            i = i + 1


    np.save(NUMPY_SAVED_IMAGES, x)
    np.save(NUMPY_SAVED_SCORES, y)
