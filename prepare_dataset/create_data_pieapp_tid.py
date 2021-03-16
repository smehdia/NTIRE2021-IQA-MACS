import cv2
import h5py
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split

'''
This code converts all images in the PIEAPP, TID 2008 and TID 2013 into
 two H5 files (one for validation and one for training)
 Each H5 file has two components ["x"] and ["y"], where ["y"] is an (N, 1) array which contains the scores
 and ["x"] is an (N, 2, IMG_SIZE, IMG_SIZE, NUM_CHANNELS) array in which [N, 0] is the Nth Reference Image and 
 [N,1] Nth Distorted Image  
'''

H5_dataset_training_name = 'data_train.h5'
H5_dataset_validation_name = 'data_val.h5'
IMG_SIZE = 288  # Pad or Crop all of the images to be the same size (288 * 288)
NUM_CHANNELS = 3
CHUNK_SIZE = 10000  # Save data in Chunks (If your memory is not enough make this lower)
VALIDATION_RATIO = 0.2

# We Standardize the scores of each distorted image, here is the mean and std for each of the datasets
MEAN_PIE, STD_PIE = 0.47, 0.26
MEAN_TID, STD_TID = 4.47, 1.30

# PieAPP Dataset Directory
PIEAPP_DIR = '/home/mehdi/Desktop/ntire/data/PieAPPdataset'
TID2008_DIR = '/home/mehdi/Desktop/ntire/data/tid2008'
TID2013_DIR = '/home/mehdi/Desktop/ntire/data/tid2013'


def initialize_datasets():
    with h5py.File(H5_dataset_training_name, 'w') as hf:
        hf.create_dataset("x", data=np.zeros(shape=(1, 2, IMG_SIZE, IMG_SIZE, NUM_CHANNELS), dtype='uint8'),
                          dtype='uint8',
                          maxshape=(None, None, None, None, None), compression="gzip")
        hf.create_dataset("y", data=np.zeros(shape=(1, 1), dtype='float32'), dtype='float32', maxshape=(None, None),
                          compression="gzip")

    with h5py.File(H5_dataset_validation_name, 'w') as hf:
        hf.create_dataset("x", data=np.zeros(shape=(1, 2, IMG_SIZE, IMG_SIZE, NUM_CHANNELS), dtype='uint8'),
                          dtype='uint8',
                          maxshape=(None, None, None, None, None), compression="gzip")
        hf.create_dataset("y", data=np.zeros(shape=(1, 1), dtype='float32'), dtype='float32', maxshape=(None, None),
                          compression="gzip")

    return


def write_in_file(path, x, y):
    with h5py.File(path, 'a') as hf:
        print(x.shape)
        print(y.shape)
        hf["x"].resize((hf["x"].shape[0] + x.shape[0]), axis=0)
        hf["x"][-x.shape[0]:] = x
        hf["y"].resize((hf["y"].shape[0] + y.shape[0]), axis=0)
        hf["y"][-y.shape[0]:] = y
        print(hf['x'].shape)
        print('Data Has been saved')

    return


def save_and_process_pieapp_dataset():
    images = []
    scores = []

    for d in ['train', 'val']:
        images_list_path = PIEAPP_DIR + '/{}_reference_list.txt'.format(d)
        images_list = open(images_list_path, 'r').readlines()
        for ref_img_name in tqdm(images_list):
            ref_info_csv_path = PIEAPP_DIR + '/labels/{}/{}_pairwise_labels.csv'.format(d,
                                                                                        ref_img_name.split(
                                                                                            '.')[0])
            info_ref = pd.read_csv(ref_info_csv_path)
            ref_image_path = PIEAPP_DIR + '/reference_images/{}/{}'.format(d, ref_img_name)
            ref_image_path = ref_image_path.rstrip()
            for i in range(info_ref[' distorted image A'].shape[0]):
                distorted_img_path = PIEAPP_DIR + '/distorted_images/{}/{}/{}'.format(d,
                                                                                      ref_img_name.split('.')[0],
                                                                                      info_ref[
                                                                                          ' distorted image A'][
                                                                                          i])
                ref_img = cv2.imread(ref_image_path)
                distorted_img = cv2.imread(distorted_img_path)
                # Pad PieAPP images to be 288 * 288
                ref_img_zero_padded = np.zeros(shape=(IMG_SIZE, IMG_SIZE, 3), dtype='uint8')
                ref_img_zero_padded[144 - 128:144 + 128, 144 - 128:144 + 128, :] = ref_img
                distorted_img_zero_padded = np.zeros(shape=(IMG_SIZE, IMG_SIZE, 3), dtype='uint8')
                distorted_img_zero_padded[144 - 128:144 + 128, 144 - 128:144 + 128, :] = distorted_img
                ref_img = ref_img_zero_padded
                distorted_img = distorted_img_zero_padded

                score = info_ref[' processed preference for A'][i]
                # Standardize the score
                score = (score - MEAN_PIE) / STD_PIE

                images.append([ref_img, distorted_img])
                scores.extend([score])
                if len(scores) % 1000 == 0:
                    print(len(scores))
                if len(scores) >= CHUNK_SIZE:
                    images = np.array(images)
                    scores = np.array(scores).reshape([-1, 1])
                    images_train, images_val, y_train, y_val = train_test_split(images, scores, test_size=VALIDATION_RATIO)
                    write_in_file('data_train.h5', images_train, y_train)
                    write_in_file('data_val.h5', images_val, y_val)
                    images = []
                    scores = []

    if len(images) != 0:
        images = np.array(images)
        scores = np.array(scores).reshape([-1, 1])
        images_train, images_val, y_train, y_val = train_test_split(images, scores, test_size=VALIDATION_RATIO)
        write_in_file('data_train.h5', images_train, y_train)
        write_in_file('data_val.h5', images_val, y_val)

    return


def save_and_process_tid_datasets(flag_tid2008):
    if flag_tid2008:
        dataset_dir = TID2008_DIR
    else:
        dataset_dir = TID2013_DIR

    images = []
    scores = []
    scores_info = open(dataset_dir + '/mos_with_names.txt', 'r').readlines()
    distorted_addresses = glob(dataset_dir + '/distorted_images/*')

    scores_info = list(map(lambda x: x.rstrip().lower(), scores_info))

    for dist_address in tqdm(distorted_addresses):
        dist_img = cv2.imread(dist_address)
        ref_address = dataset_dir + '/reference_images/' + dist_address.split('/')[-1].split('_')[
            0].upper() + '.BMP'

        dist_name = dist_address.split('/')[-1].lower()
        score = float(list(filter(lambda x: dist_name in x, scores_info))[0].split(' ')[0])
        score = (score - MEAN_TID) / STD_TID

        ref_img = cv2.imread(ref_address)
        center_x, center_y = ref_img.shape[1] // 2, ref_img.shape[0] // 2

        ref_img = ref_img[center_y - 144:center_y + 144, center_x - 144: center_x + 144]
        dist_img = dist_img[center_y - 144:center_y + 144, center_x - 144: center_x + 144]

        scores.extend([score])

        images.append([ref_img, dist_img])

        if len(scores) % 1000 == 0:
            print(len(scores))
        if len(scores) >= CHUNK_SIZE:
            images = np.array(images)
            scores = np.array(scores).reshape([-1, 1])

            images = np.array(images)
            scores = np.array(scores).reshape([-1, 1])
            images_train, images_val, y_train, y_val = train_test_split(images, scores, test_size=VALIDATION_RATIO)
            write_in_file(H5_dataset_training_name, images_train, y_train)
            write_in_file(H5_dataset_validation_name, images_val, y_val)
            images = []
            scores = []

    if len(images) != 0:
        images = np.array(images)
        scores = np.array(scores).reshape([-1, 1])
        images_train, images_val, y_train, y_val = train_test_split(images, scores, test_size=VALIDATION_RATIO)
        write_in_file(H5_dataset_training_name, images_train, y_train)
        write_in_file(H5_dataset_validation_name, images_val, y_val)

    return


if __name__ == "__main__":
    initialize_datasets()
    save_and_process_pieapp_dataset()  # PieAPP Dataset
    save_and_process_tid_datasets(False)  # TID 2008 Dataset
    save_and_process_tid_datasets(True)  # TID 2013 Dataset
