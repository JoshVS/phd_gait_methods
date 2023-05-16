from dataset import NaiveKinectDataset
from naive_classifier import create_classifier
import wandb
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt


from scipy.signal import savgol_filter, find_peaks
import numpy as np
from scipy.optimize import curve_fit


ds = NaiveKinectDataset(max_samples=20)