# -*- coding: utf-8 -*-
"""
Neural network deals with a snake game 

Created on June 19 13:44:13 2019

Python 3.7. 

@author: M.Cadek
"""

from snake import Snake
import numpy as np
from tqdm import tqdm

from tflearn import DNN
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression

def getTrainingData(_gameNumbers, gui_ = False):
    _input_data = []
    _output_data = []
    _gameNumbers = int(_gameNumbers)
    for i in tqdm(range(_gameNumbers)):
        s = Snake(gui=gui_)

        input_vect, output_vect = s.play()
        _input_data.append(input_vect)
        _output_data.append(output_vect)

    return _input_data, _output_data


def testNetworkTF(filename, gamesNum, _gui=True, model=None):
    _input_data = []
    _output_data = []
    _score = []

    s2 = Snake(gui=_gui)

    print("--- loading model ---")
    # _model.load(filename)
    if model != None:
        _model = model
    else:
        _model = create_modelTF()
#         _model.load("D:/Python3/snake_NN/snake_nn.tfl", weights_only=True)
        _model.load("D:\\Python3\\snake_NN2\\"+filename, weights_only=True)

    n = 0
    while n < gamesNum:
        input_vect, output_vect, score = s2.play(testNN=True, _model=_model)
        print(input_vect)
        print(output_vect)
        _input_data.extend(input_vect)
        _output_data.extend(output_vect)
        _score.append(score)
        n += 1

    return _input_data, _output_data, _score



def trainNetworkTF(x, y, model, filename):
    
    X = np.array([i[0] for i in x]).reshape(-1, 5, 1)
    Y = np.array([i[0] for i in y]).reshape(-1, 1)
    model.fit(X, Y, n_epoch=3, shuffle=True, run_id=filename)
    print("--- saving trained model ---")
    model.save(filename)
    return model


def create_modelTF():
    network = input_data(shape=[None, 5, 1], name='input')
    network = fully_connected(network, 25, activation='relu')
    network = fully_connected(network, 1, activation='linear')
    network = regression(network, optimizer='adam', learning_rate=1e-2, loss='mean_square', name='target')
#     model = DNN(network, checkpoint_path='snake_nn.tfl', tensorboard_dir='log', max_checkpoints=1 )
    model = DNN(network, tensorboard_dir='log')
    return model


def main():
    inputData, outputData, score = getTrainingData(1000, gui_=False)
    modelTF = create_modelTF()

    NN_fileName = "snake_nn.tfl"

    trained_model = trainNetworkTF(inputData, outputData, modelTF, NN_fileName)
    #inputTestData, outputTestData, scoreTest = testNetworkTF(NN_fileName, 50, False)
    #print("max score: ", max(scoreTest))
    #print(len(inputTestData))
    #print(len(outputTestData))


if __name__ == "__main__":
    main()
