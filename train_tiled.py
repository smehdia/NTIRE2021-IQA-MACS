import math
from tensorflow.keras.optimizers import Adam
from metrics_losses.metrics_and_losses import *
from model_tiled import get_model_tiled
from data_generators.DataGenerator_pipal import DataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from data_generators.DataGenerator_h5 import DataGeneratorH5, DataGeneratorValH5


# *********  Setting part  (replace that with suitable setting from configs.txt) ********

configs = {'MODEL_NAME': "./models/model_tiled.hdf5", 'VALIDATION_RATIO': 0.2,
           'TOTAL_NUM_TRAINING': 37120, 'TOTAL_NUM_VALIDATION': 9280,
           'learning_rate': 1e-4, 'epochs': 30, 'batch_size': 16, 'alpha': 0.5,
           'nfs': [8, 16, 32, 64, 64, 64, 32, 16, 8, 32], 'kss': [5, 5, 3, 3],
           'dense_num': 32, 'l2_reg': [1e-6, 1e-4], 'mse_factor': 0.2, 'spearman_factor': 0.1,
           'num_channels': 3, 'learning_rate_drop': 0.1, 'drop_out': 0.0, 'drop_learning_rate_after_epochs': 15.0,
           'tile_size': 72, 'lstm_units': 32}

# PIPAL data generator
training_generator = DataGenerator(False, configs['batch_size'])
validation_generator = DataGenerator(True, configs['batch_size'])
pretrained_weights = './pretrained_models/model_tiled_pretrained_h5.hdf5'
use_pretrained_weights = True

# ********* End of Setting part  (replace that with suitable setting from configs.txt) ********


def step_decay(epoch):
    initial_lrate = configs['learning_rate']
    drop = configs['learning_rate_drop']
    epochs_drop = configs['drop_learning_rate_after_epochs']
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


if __name__ == "__main__":
    model = get_model_tiled(configs['num_channels'], configs['nfs'], configs['kss'], configs['l2_reg'], configs['alpha'],
                      configs['drop_out'],
                      configs['dense_num'], configs['tile_size'], configs['lstm_units'])
    if use_pretrained_weights:
        model.load_weights(pretrained_weights)
    optimizer = Adam(configs['learning_rate'])
    checkpoint = ModelCheckpoint(configs['MODEL_NAME'], monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    lrate = LearningRateScheduler(step_decay, verbose=1)
    model.compile(loss=Final_loss(configs['mse_factor'], configs['spearman_factor']), optimizer=optimizer,
                  metrics=[correlation_coefficient_loss, total_score])
    model.fit(training_generator, validation_data=validation_generator, epochs=configs['epochs'],
              callbacks=[checkpoint, lrate], steps_per_epoch=configs['TOTAL_NUM_TRAINING'] // configs['batch_size'],
              validation_steps=configs['TOTAL_NUM_VALIDATION'] // configs['batch_size'], workers=1)
