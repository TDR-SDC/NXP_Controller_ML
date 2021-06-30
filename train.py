import tensorflow as tf
import numpy as np
from model import get_model
import glob


def train_model(model, input_shape, EPOCHS = 40)
    WIDTH = input_shape[0]
    HEIGHT = input_shape[1]
    for e in range(EPOCHS):
        FILE_I_END = glob.glob(r'Training_Data\training_data_*.npy')
        data_order = [i for i in range(1, FILE_I_END + 1)]
        shuffle(data_order)
        for count, i in enumerate(data_order):
            try:   #todo remove for debugging
                file_name = r'Training_Data\training_data_{}.npy'.format(i)
                train_data = np.load(file_name)
                print('training_data_{}.npy'.format(i), len(train_data))
                test_split = 0.1
                train_len = len(train_data) * (1-test_split)

                train = train_data[:train_len]
                test = train_data[train_len:]

                X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 3)
                Y = [i[1] for i in train] #todo reshape(-1, shape) if doesn't work

                test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,3)
                test_y = [i[1] for i in test]

                history = model.fit({'input': X}, {'targets': Y}, n_epoch=1,
                                    validation_set=({'input': test_x}, {'targets': test_y}),)
                                                                #todo (inputs=inputs, outputs=outputs) if doesn't work
                #todo implement early stopping
                if count%10 == 0:
                print('SAVING MODEL!')
                model.save(('Weights_{}').format(count))


                del train_data

            except:
                assert("File not Found")


if __name__ == '__main__':
    input_shape = (216,216,3)
    pool_size = (2,2)
    model = get_model(input_shape, pool_size)
    train_model(model, input_shape, 40)
