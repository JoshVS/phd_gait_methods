from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Softmax, Activation, Dropout, RNN, GRU
from tensorflow.keras.activations import tanh
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall, FalsePositives, FalseNegatives, TruePositives, TrueNegatives
from tensorflow.keras.optimizers import Adam

# N = 666

TRACKING_METRICS = [
    # CategoricalAccuracy(),
    # Precision(),
    Recall(),
    FalsePositives(),
    FalseNegatives(),
    TruePositives(),
    TrueNegatives()
]

def create_transformer(dataset, lr, loss):
    n_classes = dataset.n_classes

    transformer = Sequential()


def create_classifier(dataset, lr, loss):
    n_classes = dataset.n_classes
    input_shape = dataset.X.shape[1:]
    N = input_shape[-1]
    gait_recognition_model = Sequential()
    gait_recognition_model.add(LSTM(N, input_shape=input_shape, return_sequences=True))
    gait_recognition_model.add(BatchNormalization())
    gait_recognition_model.add(LSTM(N *2, input_shape=input_shape, return_sequences=True))
    gait_recognition_model.add(BatchNormalization())
    gait_recognition_model.add(LSTM(N *4, input_shape=input_shape, return_sequences=True))
    gait_recognition_model.add(BatchNormalization())
    gait_recognition_model.add(LSTM(N *8, input_shape=input_shape, return_sequences=True))
    gait_recognition_model.add(BatchNormalization())
    gait_recognition_model.add(LSTM(N *16, input_shape=input_shape))
    # gait_recognition_model.add(BatchNormalization())
    # gait_recognition_model.add(Dense(n_classes))
    # gait_recognition_model.add(BatchNormalization())
    # gait_recognition_model.add(Activation(tanh))
    gait_recognition_model.add(Dense(1024, activation='tanh'))
    gait_recognition_model.add(Dropout(0.9))
    gait_recognition_model.add(Dense(512, activation='tanh'))
    gait_recognition_model.add(Dropout(0.9))
    gait_recognition_model.add(Dense(n_classes, activation='softmax'))
    # gait_recognition_model.add(Softmax())
    gait_recognition_model.compile(
        loss=loss, 
        optimizer=Adam(learning_rate=lr),
        metrics = TRACKING_METRICS)
    # print(dir(gait_recognition_model))
    # quit()
    return gait_recognition_model

