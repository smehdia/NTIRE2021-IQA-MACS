import tensorflow as tf
from scipy.stats import spearmanr
import tensorflow.keras.backend as K
from tensorflow.keras.losses import mse
from tensorflow.keras.models import load_model

'''
The metrics and loss functions for the models
'''

RANKING_MODEL_PATH = './pretrained_models/ranking_model.hdf5'


# pearson correlation coefficient loss (1-Pearson^2)
def correlation_coefficient_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x - mx, y - my
    r_num = K.sum(tf.multiply(xm, ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return 1 - K.square(r)


# pearson score
def get_pearson_r(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x - mx, y - my
    r_num = K.sum(tf.multiply(xm, ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den
    r = K.maximum(K.minimum(r, 1.0), -1.0)

    return r


# total score (pearson + spearman)
def total_score(y_true, y_pred):
    return tf.math.add(tf.abs(get_pearson_r(y_true, y_pred)), tf.abs(get_spearman_rankcor(y_true, y_pred)))


# get spearman score
def get_spearman_rankcor(y_true, y_pred):
    return (tf.py_function(spearmanr, [tf.cast(y_pred, tf.float32),
                                       tf.cast(y_true, tf.float32)], Tout=tf.float32))


# get final loss function (mse + pearson + surrogate ranking loss)
def Final_loss(mse_factor, spearman_factor):
    ranking_model = load_model(RANKING_MODEL_PATH)
    ranking_model.trainable = False
    for layer in ranking_model.layers:
        layer.trainable = False

    def final_loss(y_true, y_pred):
        y_1 = tf.reshape(y_true, [1, 16, 1])
        y_2 = tf.reshape(y_pred, [1, 16, 1])
        a = ranking_model(y_1)
        b = ranking_model(y_2)
        mse_rank = K.mean(K.square(a - b))

        return mse_factor * mse(y_true, y_pred) + (1.0 - mse_factor) * correlation_coefficient_loss(y_true,
                                                                                                    y_pred) + spearman_factor * mse_rank

    return final_loss
