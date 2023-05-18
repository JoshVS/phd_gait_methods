from dataset import NaiveKinectDataset
from naive_classifier import create_classifier, create_frame_level_classifier
import wandb
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks

import numpy as np
from scipy.optimize import curve_fit

ds = NaiveKinectDataset(max_samples=None)
# ds.show_video(1)


epochs=10000
lr=0.001
loss='categorical_crossentropy'
# my_classifier = create_classifier(ds, lr, loss)
my_classifier = create_frame_level_classifier(ds, lr, loss)

run = wandb.init(
    project="naive_classifier",
    config={
        "learning_rate": lr,
        "epochs": epochs,
    })

class MyCallback(Callback):

    def on_epoch_end(self, epoch, logs=None):
        run.log(logs)



TRAINING_CALLBACKS = [
    MyCallback(),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, min_lr=0.0000001),
    EarlyStopping(min_delta=0.001, patience=200, monitor='loss')
]

x_frame_level = ds.X.reshape(ds.X.shape[0], ds.X.shape[1] * ds.X.shape[2])

my_classifier.fit(x_frame_level,
                  ds.y, 
                  epochs=epochs, 
                  callbacks=[TRAINING_CALLBACKS],
                  validation_split=0.3)
# simulating a training run
# my_classifier.fit(ds.X, 
#                   ds.y, 
#                   epochs=epochs, 
#                   callbacks=[TRAINING_CALLBACKS],
#                   validation_split=0.3)