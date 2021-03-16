import math
import numpy as np
from tqdm import tqdm
from scipy.stats import rankdata
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam

'''
In this script we train a model to learn sorting task, this model will be used to have an surrogate spearman loss
function
'''

TOTAL_NUM_DATA = 2000000
BATCH_SIZE = 1024
EPOCHS = 100

def create_data():
    x = np.zeros(shape=(TOTAL_NUM_DATA, 16))

    for i in range(16):
        x[0:TOTAL_NUM_DATA//2, i] = np.random.normal(0, 2, TOTAL_NUM_DATA//2)

    for i in range(16):
        x[TOTAL_NUM_DATA//2::, i] = np.random.uniform(-10, 10, TOTAL_NUM_DATA//2)

    y = np.zeros_like(x)
    for i in tqdm(range(x.shape[0])):
        y[i] = rankdata(x[i])


    return x, y


def get_model():
    inp = Input(shape=(16, 1))

    x = Conv1D(8, 2, padding='same')(inp)
    x = PReLU()(x)

    x = Conv1D(16, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    x = Conv1D(32, 5, padding='same')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    x = Conv1D(64, 7, padding='same')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    x = Conv1D(96, 10, padding='same')(x)
    x = PReLU()(x)

    x = Conv1D(128, 7, padding='same')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    x = Conv1D(256, 7, padding='same')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    x = Flatten()(x)
    x = Dense(16)(x)

    model = Model(inp, x)
    model.summary()

    return model

# learning rate schedule
def step_decay(epoch):
    initial_lrate = 1e-3
    drop = 0.1
    epochs_drop = 30.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate

if __name__ == "__main__":
    x, y = create_data()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = get_model()
    optimizer = Adam(1e-3)
    filepath = "./ranking_model.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    tbCallBack = TensorBoard(log_dir='./ranking_model', histogram_freq=0, write_graph=True,
                             write_images=True)
    lrate = LearningRateScheduler(step_decay, verbose=1)

    model.compile(loss='mae', optimizer=optimizer,
                  metrics=['mae'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS,
              callbacks=[checkpoint, tbCallBack, lrate], batch_size=BATCH_SIZE)


get_model()
