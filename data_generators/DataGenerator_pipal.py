import cv2
import math
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split

'''
PIPAL data generator, we use intensity scaling (on the L channel in Lab color space), rotations and flipping 
the images for data augmentation
'''

# pipal scores information
mu = 1449.05
std = 121.35

VALIDATION_RATIO = 0.2
NUMPY_PIPAL_DIR_IMAGES = './prepare_dataset/total_inputs.npy'
NUMPY_PIPAL_DIR_SCORES = './prepare_dataset/total_scores.npy'


def scale_down_intensity(img, scale_fac):
    l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2Lab))
    l_new = (scale_fac * l).astype('uint8')
    lab_new = cv2.merge([l_new, a, b])
    new_img = cv2.cvtColor(lab_new, cv2.COLOR_Lab2BGR)

    return new_img


class DataGenerator(Sequence):

    def __init__(self, val_flag, batch_size):
        x, y = np.load(NUMPY_PIPAL_DIR_IMAGES), np.load(NUMPY_PIPAL_DIR_SCORES)
        # Data augmentation by changing intensities
        x_aug = np.zeros_like(x)
        y_aug = np.copy(y)
        for i in tqdm(range(x.shape[0])):
            img_ref = x[i, 0]
            img_dist = x[i, 1]

            random_scale = random.uniform(0.3, 1.5)
            img_ref_new = scale_down_intensity(img_ref, random_scale)
            img_dist_new = scale_down_intensity(img_dist, random_scale)

            x_aug[i, 0] = img_ref_new
            x_aug[i, 1] = img_dist_new

        x = np.concatenate([x, x_aug], axis=0)
        y = np.concatenate([y, y_aug], axis=0)

        y = (y - mu) / (std)

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=VALIDATION_RATIO)

        if val_flag:
            self.x, self.y = x_test, y_test
        else:
            self.x, self.y = x_train, y_train
        self.val_flag = val_flag
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
                                               self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
                                               self.batch_size]

        batch_x_ref, batch_x_distorted = batch_x[:, 0], batch_x[:, 1]

        if not self.val_flag:
            random_rotation = random.choice([0, 1, 2, 3])  # 0, 90, 180, 270
            batch_x_ref = tf.image.rot90(batch_x_ref, k=random_rotation)
            batch_x_distorted = tf.image.rot90(batch_x_distorted, k=random_rotation)

            random_flip = random.choice([0, 1])
            if random_flip:
                batch_x_ref = tf.image.flip_left_right(batch_x_ref)
                batch_x_distorted = tf.image.flip_left_right(batch_x_distorted)
            else:
                batch_x_ref = batch_x_ref
                batch_x_distorted = batch_x_distorted

            random_flip = random.choice([0, 1])
            if random_flip:
                batch_x_ref = tf.image.flip_up_down(batch_x_ref)
                batch_x_distorted = tf.image.flip_up_down(batch_x_distorted)
            else:
                batch_x_ref = batch_x_ref
                batch_x_distorted = batch_x_distorted

            random_flip = random.choice([0, 1])
            if random_flip:
                batch_x_ref = tf.image.flip_up_down(batch_x_ref)
                batch_x_distorted = tf.image.flip_up_down(batch_x_distorted)
                batch_x_ref = tf.image.flip_left_right(batch_x_ref)
                batch_x_distorted = tf.image.flip_left_right(batch_x_distorted)
            else:
                batch_x_ref = batch_x_ref
                batch_x_distorted = batch_x_distorted

            batch_x_ref = batch_x_ref.numpy()
            batch_x_distorted = batch_x_distorted.numpy()

        return [batch_x_ref, batch_x_distorted], batch_y
