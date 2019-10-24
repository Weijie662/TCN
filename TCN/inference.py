import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler
from keras.optimizers import SGD, Adam
from keras.losses import squared_hinge
import os
import argparse
import keras.backend as K
from models.BuildTCNClassifier import build_model
from utils.load_data_google_dataset import load_dataset_google_dataset
import numpy as np
from keras.backend.tensorflow_backend import set_session
import sys

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
    #config.gpu_options.visible_device_list = "0"  # only the gpu 0 is allowed
set_session(tf.Session(config=config))








batch_size=256
epochs=1000
dataset='GOOGLE-DATASET'
name='lstm'
progress_logging=1
tensorboard_name = '{}_{}.hdf5'.format(dataset,name)


# ## Construct the network
print('Construct the Network\n')
model = build_model(batch_size)
print('setting up the network and creating callbacks\n')
early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=10, mode='min', verbose=1)
checkpoint = ModelCheckpoint('weights/model.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)
tensorboard = TensorBoard(log_dir='./logs/fxp_' + str(tensorboard_name), histogram_freq=0, write_graph=True, write_images=False)


## Scheduler
def scheduler(epoch):
    return K.get_value(model.optimizer.lr)
lr_decay = LearningRateScheduler(scheduler)


# Loading of Dataset
tr, v, tst, tr_y,v_y,tst_y = load_dataset_google_dataset()

# Number of times the subset is passed through training
cycles_per_subdataset=1

# Size of each subset taken for training
size_subdataset=22016
# Number of batches per subset
batches_per_subdataset=int(size_subdataset/batch_size)



#model.load_weights('weights/model_CRNN_unidirectional.hdf5', by_name=True) # Optional loading of model
model.load_weights('weights/model.hdf5', by_name=True) # Optional loading of model

a,accuracy_tst=model.evaluate(x=tst, y=tst_y, batch_size=batch_size, verbose=1)
print("Test Accuracy")
print(accuracy_tst)







