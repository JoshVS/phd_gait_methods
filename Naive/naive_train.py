from dataset import NaiveVideoDataset
from naive_classifier import create_classifier
import wandb
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt


from scipy.signal import savgol_filter, find_peaks
import numpy as np
from scipy.optimize import curve_fit

def test_fun(x, dist, amp, omega, phi):
    return dist + amp * np.cos(omega * x + phi)

skel_data = NaiveVideoDataset(max_samples=20)
d = skel_data.distances

selection = 3

x = np.arange(len(d[selection]))
params, _ = curve_fit(test_fun, x, d[selection])

plt.figure()
plt.plot(d[selection])

plt.plot([test_fun(a, params[0], params[1], params[2], params[3]) for a in x])
plt.savefig("distances/test_fun.png")

quit()

epochs=10000
lr=0.00001
loss='categorical_crossentropy'
my_classifier = create_classifier(skel_data, lr, loss)

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

# simulating a training run
my_classifier.fit(skel_data.X, 
                  skel_data.y, 
                  epochs=epochs, 
                  callbacks=[TRAINING_CALLBACKS],
                  validation_split=0.3)