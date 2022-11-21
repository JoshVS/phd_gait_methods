from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Softmax, Activation
from tensorflow.keras.activations import tanh

N = 666


def create_classifier(n_classes = 1, N=N):
    gait_recognition_model = Sequential()
    gait_recognition_model.add(LSTM(N, input_shape=(7, 666), return_sequences=True))
    gait_recognition_model.add(BatchNormalization())
    gait_recognition_model.add(LSTM(N, input_shape=(7, 666)))
    gait_recognition_model.add(Dense(n_classes))
    gait_recognition_model.add(BatchNormalization())
    gait_recognition_model.add(Activation(tanh))
    gait_recognition_model.add(Softmax())
    gait_recognition_model.compile(loss='categorical_crossentropy', optimizer='adam')
    return gait_recognition_model

