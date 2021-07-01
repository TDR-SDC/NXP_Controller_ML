import tensorflow as tf
import numpy as np
from model import get_model
import glob
from random import shuffle


def train_model(model, input_shape, EPOCHS = 40):
    WIDTH = input_shape[0]
    HEIGHT = input_shape[1]
    epoch_count = 0
    for e in range(EPOCHS):
        FILE_I_END = glob.glob(r'Training_Data\training_data_*.npy')
        data_order = [i for i in range(0, len(FILE_I_END) )]
        shuffle(data_order)
        for count, i in enumerate(data_order):
            file_name = r'Training_Data\training_data_{}.npy'.format(i)
            train_data = np.load(file_name, allow_pickle=True)
            print('training_data_{}.npy'.format(i), len(train_data))
            test_split = 0.1
            train_len = int(len(train_data) * (1-test_split))

            train = train_data[:train_len]
            test = train_data[train_len:]

            X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 3)
            Y = np.array([i[1] for i in train]).reshape(-1,2)
            # print(X.shape)
            # print(Y.shape)
            test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,3)
            test_y = np.array([i[1] for i in test]).reshape(-1,2)

            history = model.fit(x= {"input": X}, y= {'targets': Y}, epochs= 1, validation_data= ({'input': test_x}, {'targets': test_y}) ) #validation_data=({'input': test_x}, {'targets': test_y})

            epoch_count += 1
            print(epoch_count)
            #todo implement early stopping
            if epoch_count % 10 == 0:
                print('SAVING MODEL!')
                model.save(('Weights\Weights_{}').format(epoch_count))

            del train_data
            print('Done')



if __name__ == '__main__':
    input_shape = (216,216,3)
    pool_size = (2,2)
    model = get_model(input_shape, pool_size)
    train_model(model, input_shape, 40)
