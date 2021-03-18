import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

'''
model architecture with tiling the input and ConvLSTM
'''


def tile_image(inp, tile_size):
    tiles = []
    inp_ten = tf.expand_dims(inp, axis=1)
    for y in range(0, 288, tile_size):
        for x in range(0, 288, tile_size):
            tiles.append(inp_ten[:, :, y:y + tile_size, x:x + tile_size, :])

    tiles = tf.concat(tiles, axis=1)

    return tiles


def normalize_tensor_image(inp):
    out = tf.convert_to_tensor(inp)
    out = tf.dtypes.cast(out, tf.float32)
    out = (out - 127.5) / 127.5
    return out


def get_shared_model(nc, nfs, kss, l2regconv, alpha, tile_size, conv_lstm):
    kss1, kss2, kss3, kss4 = kss
    nf1, nf2, nf3, nf4 = nfs[0], nfs[1], nfs[2], nfs[3]
    l2reg1 = tf.keras.regularizers.l2(l2regconv)
    inp_tensor = Input(shape=(int(288 / tile_size) * int(288 / tile_size), tile_size, tile_size, nc))

    first_lstm = ConvLSTM2D(conv_lstm, (5, 5), padding='same')(inp_tensor)

    first_conv = Conv2D(3, (kss1, kss1), padding='same', kernel_regularizer=l2reg1)(first_lstm)

    y1 = Conv2D(nf1, (kss1, kss1), padding='same', kernel_regularizer=l2reg1)(first_conv)
    y1 = LeakyReLU(alpha)(y1)

    y2 = MaxPooling2D(2)(y1)
    y2 = Conv2D(nf2, (kss2, kss2), padding='same', kernel_regularizer=l2reg1)(y2)
    y2 = LeakyReLU(alpha)(y2)

    y3 = MaxPooling2D(2)(y2)
    y3 = Conv2D(nf3, (kss3, kss3), padding='same', kernel_regularizer=l2reg1)(y3)
    y3 = LeakyReLU(alpha)(y3)

    y4 = MaxPooling2D(2)(y3)
    y4 = Conv2D(nf4, (kss4, kss4), padding='same', kernel_regularizer=l2reg1)(y4)
    y4 = LeakyReLU(alpha)(y4)

    model = Model(inp_tensor, [y1, y2, y3, y4])

    model.summary()

    return model


def get_model_tiled(num_channels, nfs, kss, l2regfactors, alpha, dropout_factor, num_dense, tile_size, lstm_units):
    inp_tensor1 = Input(shape=(288, 288, num_channels))
    inp_tensor2 = Input(shape=(288, 288, num_channels))

    n_img1 = normalize_tensor_image(inp_tensor1)
    n_img2 = normalize_tensor_image(inp_tensor2)

    total_inp_1 = tile_image(n_img1, tile_size)
    total_inp_2 = tile_image(n_img2, tile_size)

    l2reg_dense, l2regconv = l2regfactors

    shared_model = get_shared_model(total_inp_1.shape[-1], nfs, kss, l2regconv, alpha, tile_size, lstm_units)

    [y1_1, y2_1, y3_1, y4_1] = shared_model(total_inp_1)
    [y1_2, y2_2, y3_2, y4_2] = shared_model(total_inp_2)

    branch = Conv2D(nfs[4], (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(l2regconv))(y4_2)
    branch = LeakyReLU(alpha)(branch)

    diff_1 = tf.math.abs(y4_1 - y4_2)

    branch = Concatenate(axis=-1)([branch, diff_1])

    branch = Conv2D(nfs[5], (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(l2regconv))(branch)
    branch = LeakyReLU(alpha)(branch)
    branch = UpSampling2D(2)(branch)

    diff_2 = tf.math.abs(y3_1 - y3_2)
    branch = Concatenate(axis=-1)([branch, diff_2])

    branch = Conv2D(nfs[6], (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(l2regconv))(branch)
    branch = LeakyReLU(alpha)(branch)
    branch = UpSampling2D(2)(branch)

    diff_3 = tf.math.abs(y2_1 - y2_2)

    branch = Concatenate(axis=-1)([branch, diff_3])

    branch = Conv2D(nfs[7], (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(l2regconv))(branch)
    branch = LeakyReLU(alpha)(branch)
    branch = UpSampling2D(2)(branch)

    diff_4 = tf.math.abs(y1_1 - y1_2)

    branch = Concatenate(axis=-1)([branch, diff_4])

    branch = Conv2D(nfs[8], (3, 3), padding='same', strides=(4, 4),
                    kernel_regularizer=tf.keras.regularizers.l2(l2regconv))(branch)
    branch = LeakyReLU(alpha)(branch)
    branch = Conv2D(nfs[9], (3, 3), padding='same', strides=(2, 2),
                    kernel_regularizer=tf.keras.regularizers.l2(l2regconv))(branch)
    branch = LeakyReLU(alpha)(branch)

    branch = Flatten()(branch)

    branch = Dense(num_dense, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2reg_dense))(branch)
    branch = Dropout(dropout_factor)(branch)

    branch = Dense(1, activation='sigmoid')(branch)

    branch = Dense(1)(branch)

    model = Model([inp_tensor1, inp_tensor2], branch)
    # tf.keras.utils.plot_model(
    #     model, to_file='model.png',
    #     expand_nested=True)
    model.summary()

    return model
