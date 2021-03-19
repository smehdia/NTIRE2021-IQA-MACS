import math
from model_simple import get_model
from tensorflow.keras.optimizers import Adam
from metrics_losses.metrics_and_losses import *
from data_generators.DataGenerator_pipal import DataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from data_generators.DataGenerator_h5 import DataGeneratorH5, DataGeneratorValH5

'''
This code train the models (without attention and batch normalization)
'''

# *********  Setting part  (replace that with suitable setting from configs.txt) ********


configs = {'MODEL_NAME': "./pretrained_models/model1_pretrained_h5.hdf5", 'VALIDATION_RATIO': 0.2,
           'TOTAL_NUM_TRAINING': 65585, 'TOTAL_NUM_VALIDATION': 16397,
           'learning_rate': 1e-4, 'epochs': 25, 'batch_size': 16, 'alpha': 0.7,
           'nfs': [8, 16, 32, 64, 64, 64, 32, 16, 8, 16], 'kss': [5, 5, 3, 3],
           'dense_num': 32, 'l2_reg': [0, 1e-2], 'mse_factor': 0.1, 'spearman_factor': 0.1,
           'num_channels': 3, 'learning_rate_drop': 0.1, 'drop_out': 0.0, 'drop_learning_rate_after_epochs': 10.0}
# If you want to pretrain the network set these as H5 otherwise comment H5 generators and uncomment the Pipal generators
# training_generator = DataGeneratorH5(configs['batch_size'])
# validation_generator = DataGeneratorValH5(configs['batch_size'])
# PIPAL data generator
training_generator = DataGenerator(False, configs['batch_size'])
validation_generator = DataGenerator(True, configs['batch_size'])
pretrained_weights = './pretrained_models/model1_pretrained_h1.hdf5'
use_pretrained_weights = False


# ********* End of Setting part  (replace that with suitable setting from configs.txt) ********


def step_decay(epoch):
    initial_lrate = configs['learning_rate']
    drop = configs['learning_rate_drop']
    epochs_drop = configs['drop_learning_rate_after_epochs']
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


if __name__ == "__main__":
    model = get_model(configs['num_channels'], configs['nfs'], configs['kss'], configs['l2_reg'], configs['alpha'],
                      configs['drop_out'],
                      configs['dense_num'])
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
