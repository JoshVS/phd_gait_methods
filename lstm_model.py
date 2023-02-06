from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Softmax, Activation
from tensorflow.keras.activations import tanh
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall
from tensorflow.keras.optimizers import Adam

# N = 666


def create_classifier(n_classes = 1, input_shape=(7, 666)):
    N = input_shape[-1]
    gait_recognition_model = Sequential()
    gait_recognition_model.add(LSTM(N, input_shape=input_shape, return_sequences=True))
    gait_recognition_model.add(BatchNormalization())
    gait_recognition_model.add(LSTM(N, input_shape=input_shape))
    gait_recognition_model.add(Dense(n_classes))
    gait_recognition_model.add(BatchNormalization())
    gait_recognition_model.add(Activation(tanh))
    gait_recognition_model.add(Softmax())
    gait_recognition_model.compile(
        loss='categorical_crossentropy', 
        optimizer=Adam(learning_rate=1e-4),
        metrics = [CategoricalAccuracy(), Precision(), Recall()])
    return gait_recognition_model

