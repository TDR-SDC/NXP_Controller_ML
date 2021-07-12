# todo Retrain the model

from numpy.core.records import array
from numpy.lib.npyio import save
import tensorflow as tf
import numpy as np
from model import get_model
import glob
from random import shuffle


def train_model(model, input_shape, EPOCHS = 40):
    WIDTH = input_shape[0]
    HEIGHT = input_shape[1]
    epoch_count = 0
    loss_hist = []
    val_loss_hist = []
    

    print("block-1")
    FILE_I_END = glob.glob(r'data/training_data_*.npy')
    print(FILE_I_END)
    for e in range(EPOCHS):
        data_order = [i for i in range(0, len(FILE_I_END))]
        shuffle(data_order)
        print(data_order)
        for i in data_order:
            file_name = r'data/training_data_{}.npy'.format(i)
            train_data = np.load(file_name, allow_pickle=True)
            print('training_data_{}.npy'.format(i), len(train_data))
            test_split = 0.1
            train_len = int(len(train_data) * (1-test_split))

            train = train_data[:train_len]
            test = train_data[train_len:]

            X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 3)
            Y = np.array([i[1] for i in train]).reshape(-1,1)
            # print(X.shape)
            # print(Y.shape)
            test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,3)
            test_y = np.array([i[1] for i in test]).reshape(-1,1)

            history = model.fit(x= {"input": X}, y= {'targets': Y}, epochs= 1, validation_data= ({'input': test_x}, {'targets': test_y}) ) #validation_data=({'input': test_x}, {'targets': test_y})
            val_loss_hist.append(history.history['val_loss'][0])
            loss_hist.append(history.history['loss'][0])

            del train_data
        
        #todo implement early stopping
        if epoch_count % 1 == 0 and epoch_count != 0:
            print('SAVING MODEL!')
            model.save(('weights/Weights_{}').format(epoch_count))
            val_loss_hist_np = np.array(val_loss_hist)
            loss_hist_np = np.array(loss_hist)
            np.save("val_loss.npy", val_loss_hist_np)
            np.save("loss_hist.npy", loss_hist_np)
            del val_loss_hist_np
            del loss_hist_np


        epoch_count += 1
        print("********EPOCH-{}**********".format(epoch_count))

if __name__ == '__main__':
    input_shape = (216,216,3)
    pool_size = (2,2)
    model = get_model(input_shape, pool_size)
    try:
        # Transfer Learning
        model = tf.keras.models.load_model("weights/Weights_6-old")
        print("************************Model Weights LOADED*****************")
    except:
        print("Prev model not available")
    print("model loaded")
    train_model(model, input_shape, 40)
    print("training")
