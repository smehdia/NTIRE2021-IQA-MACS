import cv2
import imutils
import numpy as np
from tqdm import tqdm
from glob import glob
from metrics_losses.metrics_and_losses import *

'''
evaluate the images in a directory and log the outputs
'''

IMG_FORMATS = '.bmp'
DISTORTED_IMAGES_DIR = '../../data/PIPAL_Dataset-Testing_Distorted_Ima/Dis'
REFERENCE_IMAGES_DIR = '../../data/PIPAL_Dataset-Testing_Reference_Ima/Ref'
# DISTORTED_IMAGES_DIR = '../../data/Distortion_validation/Dis'
# REFERENCE_IMAGES_DIR = '../../data/Validation_Ref/Ref'
LOG_FILE = "output.txt"
MODELS_DIR = './models'
BATCH_SIZE_PREDICTION = 1024


def avg(myArray, N=2):
    cum = np.cumsum(myArray, 0)
    result = cum[N - 1::N] / float(N)
    result[1:] = result[1:] - result[:-1]

    remainder = myArray.shape[0] % N
    if remainder != 0:
        if remainder < myArray.shape[0]:
            lastAvg = (cum[-1] - cum[-1 - remainder]) / float(remainder)
        else:
            lastAvg = cum[-1] / float(remainder)
        result = np.vstack([result, lastAvg])

    return result


def scale_down_intensity(img, scale_fac):
    l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2Lab))
    l_new = (scale_fac * l).astype('uint8')
    lab_new = cv2.merge([l_new, a, b])
    new_img = cv2.cvtColor(lab_new, cv2.COLOR_Lab2BGR)

    return new_img


def augmente_images(ref_img, distorted_img, aug_list):
    augmented_images = np.zeros(shape=(len(aug_list), 2, 288, 288, 3), dtype='uint8')
    for i, rotation in enumerate(aug_list):

        if rotation == 'flip1':
            rotated_distorted = cv2.flip(distorted_img, 0)
            rotated_ref = cv2.flip(ref_img, 0)

        elif rotation == 'flip2':
            rotated_distorted = cv2.flip(distorted_img, 1)
            rotated_ref = cv2.flip(ref_img, 1)

        elif rotation == 'flip3':
            rotated_distorted = cv2.flip(distorted_img, 0)
            rotated_ref = cv2.flip(ref_img, 0)
            rotated_distorted = cv2.flip(rotated_distorted, 1)
            rotated_ref = cv2.flip(rotated_ref, 1)

        elif 'intensity' in rotation:
            i_scale = float(rotation.split('_')[-1])
            rotated_distorted = scale_down_intensity(distorted_img, i_scale)
            rotated_ref = scale_down_intensity(ref_img, i_scale)

        else:
            angle = int(rotation)
            rotated_distorted = imutils.rotate(distorted_img, angle)
            rotated_ref = imutils.rotate(ref_img, angle)


        augmented_images[i, 0] = rotated_ref
        augmented_images[i, 1] = rotated_distorted

    return augmented_images

def get_batch_score(models, x):
    scores = np.zeros(shape=(x.shape[0], len(models), x.shape[1]))
    for j in tqdm(range(x.shape[1])):
        inps = x[:, j]
        for i in range(len(model_addresses)):
            score_model = (models[i].predict([inps[:, 0], inps[:, 1]]) * std + mu)[:, 0]
            scores[:, i, j] = score_model

    scores = np.mean(scores, axis=(1, 2))

    return scores

if __name__ == "__main__":


    # pipal scores information
    mu = 1449.05
    std = 121.35

    f = open(LOG_FILE, "w")
    intensity_scales = np.arange(0.1, 1.6, 0.1)
    aug_list = ['0', '90', '180', '270', 'flip1', 'flip2', 'flip3']
    for i in range(intensity_scales.shape[0]):
        aug_list.extend(['intensity_' + str(round(intensity_scales[i], 2))])


    val_addresses = glob(DISTORTED_IMAGES_DIR + '/*' + IMG_FORMATS)
    val_addresses = sorted(val_addresses, key=lambda x: int(
        x.split('/')[-1].split('.')[0].split('_')[0][1::] + x.split('/')[-1].split('.')[0].split('_')[1] +
        x.split('/')[-1].split('.')[0].split('_')[2]))

    model_addresses = []
    models = []
    for model_address in glob(MODELS_DIR + '/*.hdf5'):
        model_addresses.extend([model_address])
        models.extend([load_model(model_address, compile=False, custom_objects={'tf': tf})])


    total_scores = []
    x = []
    counter = 0
    for val_address in tqdm(val_addresses):
        distorted_img = cv2.imread(val_address)
        ref_img = cv2.imread(
            REFERENCE_IMAGES_DIR + '/' + val_address.split('/')[-1].split('.')[0].split('_')[0] + '.bmp')

        aug_images = augmente_images(ref_img, distorted_img, aug_list)
        x.append(aug_images)
        if len(x) == BATCH_SIZE_PREDICTION:
            batch_score = get_batch_score(models, np.array(x))
            total_scores.extend(batch_score.tolist())
            x = []

    if len(x) != 0:
        batch_score = get_batch_score(models, np.array(x))
        total_scores.extend(batch_score.tolist())
        x = []


    total_scores = np.array(total_scores).reshape([-1])
    print(total_scores.shape)
    for i in range(len(val_addresses)):
        f.write(val_addresses[i].split('/')[-1] + ',' + str(total_scores[i]) + '\n')

    f.close()

