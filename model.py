from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Flatten, Dense, Dropout
from tensorflow.keras.metrics import RootMeanSquaredError

""" Layers are Named in the following Manner " <TYPE> __ <Branch_no> - <Seq_no> " Combined branch is considered as 4th Branch"""

def get_model(input_shape=(216, 216, 3), pool_size=(2, 2), activation='relu'):
    model = Sequential(
        [
            Conv2D(16, (3, 3), input_shape=input_shape, activation=activation, name='input'),
            Conv2D(16, (3,3), activation= activation, name='Conv_1-2'),
            MaxPool2D(pool_size, name='Pool_1'),

            Conv2D(16, (3, 3), activation=activation, name='Conv_2-1'),
            Conv2D(16, (3, 3), activation=activation, name='Conv_2-2'),
            BatchNormalization(trainable=True, name='BN_1'),
            MaxPool2D(pool_size, name='Pool_2'),

            Conv2D(32, (3, 3), activation=activation, name='Conv_3-1'),
            Conv2D(32, (3, 3), activation=activation, name='Conv_3-2'),
            MaxPool2D(pool_size, name='Pool_3'),

            Conv2D(32, (3, 3), activation=activation, name='Conv_4-1'),
            Conv2D(32, (3, 3), activation=activation, name='Conv_4-2'),
            BatchNormalization(trainable=True, name='BN_2'),
            MaxPool2D(pool_size, name='Pool_4'),

            # Conv2D(64, (3, 3), activation=activation, name='Conv_5-1'),
            # Conv2D(64, (3, 3), activation=activation, name='Conv_5-2'),
            # BatchNormalization(trainable=True, name='BN_3'),
            # MaxPool2D(pool_size, name='Pool_5'),

            Flatten(name='Flatten_1'),
            Dense(1024, activation='relu', name='Dense_1'),
            Dropout(0.5, name='Drop_1'),
            Dense(512, activation='relu', name='Dense_NXP_2'),
            Dropout(0.5, name='Drop_NXP_2'),
            Dense(128, activation='tanh', name='Dense_NXP_3'),
            Dropout(0.3, name='Drop_NXP_3'),
            Dense(1, activation='tanh', name='targets')
        ]
    )

    model.compile(optimizer='adam', loss='mse', metrics=['mae', RootMeanSquaredError()])
    model.summary()

    return model

if __name__ == "__main__":
    model = get_model()
