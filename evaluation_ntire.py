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
DISTORTED_IMAGES_DIR = '../../data/Distortion_validation/Dis'
REFERENCE_IMAGES_DIR = '../../data/Validation_Ref/Ref'
LOG_FILE = "output.txt"
MODELS_DIR = './models'

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


if __name__ == "__main__":

    # pipal scores information
    mu = 1449.05
    std = 121.35

    val_addresses = glob(DISTORTED_IMAGES_DIR + '/*' + IMG_FORMATS)
    val_addresses = sorted(val_addresses, key=lambda x: int(
        x.split('/')[-1].split('.')[0].split('_')[0][1::] + x.split('/')[-1].split('.')[0].split('_')[1] +
        x.split('/')[-1].split('.')[0].split('_')[2]))


    models = []
    model_addresses = []
    for model_address in glob(MODELS_DIR + '/*.hdf5'):
        model = load_model(model_address, compile=False, custom_objects={'tf':tf})
        model.summary()
        models.extend([model])
        model_addresses.extend([model_address])


    f = open(LOG_FILE, "w")

    intensity_scales = np.arange(0.1, 1.6, 0.1)

    aug_list = ['0', '90', '180', '270', 'flip1', 'flip2', 'flip3']
    for i in range(intensity_scales.shape[0]):
        aug_list.extend(['intensity_' + str(round(intensity_scales[i], 2))])

    x = np.zeros(shape=(len(aug_list) * len(val_addresses), 2, 288, 288, 3), dtype='uint8')

    counter = 0
    for val_address in tqdm(val_addresses):
        distorted_img = cv2.imread(val_address)
        ref_img = cv2.imread(
            REFERENCE_IMAGES_DIR + '/' + val_address.split('/')[-1].split('.')[0].split('_')[0] + '.bmp')

        scores = []
        for rotation in aug_list:

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

            x[counter, 0] = rotated_ref
            x[counter, 1] = rotated_distorted

            counter += 1


    total_scores = np.zeros(shape=(len(val_addresses), len(models)))
    for i in tqdm(range(len(models))):
        scores_models = models[i].predict([x[:, 0], x[:, 1]]) * std + mu
        outs = scores_models
        outs = outs.reshape([-1])
        avg_score = avg(outs, len(aug_list))
        total_scores[:, i] = avg_score




    agg_score = np.average(total_scores, axis=1)

    for i in range(len(val_addresses)):
        f.write(val_addresses[i].split('/')[-1] + ',' + str(agg_score[i]) + '\n')
