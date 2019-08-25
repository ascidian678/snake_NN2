# from __future__ import absolute_import, division, print_function

from snake import Snake
import numpy as np

import tflearn
import math
from tflearn import DNN
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import model_from_json


# from statistics import mean
# from collections import Counter


def getTrainingData(_gameNumbers, gui_ = False):
    _input_data = []
    _output_data = []
    _gameNumbers = int(_gameNumbers)
    checklist = [x * 1000 for x in range(1, 16, 1)]

    for i in range(_gameNumbers):
        s = Snake(gui=gui_)

        input_vect, output_vect = s.play()

        if i in checklist:
            print('training progrees: ', i)

        _input_data.append(input_vect)
        _output_data.append(output_vect)

    return _input_data, _output_data


def testNetworkTF(filename, model=None):
    _input_data = []
    _output_data = []

    s2 = Snake(gui=True)

    print("--- loading model ---")
    # _model.load(filename)
    if model != None:
        _model = model
    else:
        _model = create_modelTF()
        _model.load("D:\\Python3\\snake_NN\\snake_nn.tfl", weights_only=True)

    input_vect, output_vect = s2.play(testNN=True, _model=_model)

    _input_data.append(input_vect)
    _output_data.append(output_vect)

    return _input_data, _output_data


def testNetworkKERAS(filename, model=None):
    _input_data = []
    _output_data = []

    s2 = Snake(gui=True)
    # if model != None:
    #     _model = model
    _model = create_modelKERAS()
    json_file = open('model.json', 'r')
    loaded_json_model = json_file.read()
    model = model_from_json(loaded_json_model)
    model.load_weights('model.h5')
    input_vect, output_vect = s2.play(testNN=True, _model=_model)

    _input_data.append(input_vect)
    _output_data.append(output_vect)

    return _input_data, _output_data


def trainNetworkTF(x, y, model, filename):
    # x = np.array(x).reshape(-1, 7)
    # y = np.array(y).reshape(-1, 3)
    # print(50 * "*")
    # print(x[0])
    # print(50 * "*")
    # print(y[0])
    X = np.array([i[0] for i in x]).reshape(-1, 7, 1)
    Y = np.array([i[0] for i in y]).reshape(-1, 3)
    model.fit(X, Y, n_epoch=1, shuffle=True, run_id=filename)
    print("--- saving trained model ---")
    model.save(filename)
    return model


def trainNetworkKERAS(x, y, model, filename):
    model.fit(np.array(x[0]).reshape(-1, 7), np.array(y[0]).reshape(-1, 3), epochs=6, batch_size=256)
    model.save_weights('model.h5')
    model_json = model.to_json()
    with open('model.json', 'w') as json_file:
        json_file.write(model_json)
    return model


def create_modelTF():
    network = input_data(shape=[None, 7, 1], name='input')
    network = fully_connected(network, 25, activation='relu')
    network = fully_connected(network, 3, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=1e-2, loss='mean_square', name='target')
    model = DNN(network, checkpoint_path='snake_nn.tfl', max_checkpoints=1, )
    return model


def create_modelKERAS():
    model = Sequential()
    model.add(Dense(units=9, input_dim=7))
    model.add(Dense(units=15, activation='relu'))
    model.add(Dense(units=3, activation='softmax'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model


def main():
    # inputData, outputData = getTrainingData(15000, gui_=False)
    # model = create_modelTF()
    # model = create_modelKERAS()
    NN_fileName = "snake_nn.tfl"
    # trained_model = trainNetworkTF(inputData, outputData, model, NN_fileName)
    # trained_model = trainNetworkKERAS(inputData, outputData, model, NN_fileName)

    # testNetworkTF(filename=NN_fileName)
    testNetworkKERAS(filename=NN_fileName)


if __name__ == "__main__":
    main()
