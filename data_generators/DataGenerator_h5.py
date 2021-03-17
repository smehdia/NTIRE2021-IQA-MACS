import h5py
import random
import tensorflow as tf
from tensorflow.keras.utils import Sequence

'''
Data generator for PieAPP and TID 2008, 2013 datasets, we use horizontal and vertical flips
and 90, 180, 270 rotations of the images for data augmentation
'''


H5_TRAINING_DIR = './prepare_dataset/data_train.h5'
H5_VALIDATION_DIR = './prepare_dataset/data_val.h5'

f = h5py.File(H5_TRAINING_DIR, 'r')
num_data_train = f['x'].shape[0]
g = h5py.File(H5_VALIDATION_DIR, 'r')
num_data_val = g['x'].shape[0]

class DataGeneratorH5(Sequence):

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.x = f['x']
        self.y = f['y']

    def __len__(self):
        return int(num_data_train / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
                                               self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
                                               self.batch_size]

        batch_x_ref, batch_x_distorted = batch_x[:, 0], batch_x[:, 1]

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

        batch_x_ref = batch_x_ref.numpy()
        batch_x_distorted = batch_x_distorted.numpy()

        return [batch_x_ref, batch_x_distorted], batch_y


class DataGeneratorValH5(Sequence):

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.x = g['x']
        self.y = g['y']

    def __len__(self):
        return int(num_data_val / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
                                               self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
                                               self.batch_size]

        batch_x_ref, batch_x_distorted = batch_x[:, 0], batch_x[:, 1]

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

        batch_x_ref = batch_x_ref.numpy()
        batch_x_distorted = batch_x_distorted.numpy()

        return [batch_x_ref, batch_x_distorted], batch_y
