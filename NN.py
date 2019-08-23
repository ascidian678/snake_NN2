from snake import Snake
import numpy as np

import tflearn
import math
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression
# from statistics import mean
# from collections import Counter

def getTrainingData(_gameNumbers):
	_input_data = []
	_output_data = []

	for i in range(_gameNumbers):
		
		s = Snake(gui = False)
		input_vect, output_vect = s.play()
		
		_input_data.append(input_vect)
		_output_data.append(output_vect)
		
	return _input_data, _output_data


def testNetwork(_model):

	_input_data = []
	_output_data = []
		
	s = Snake(gui = True)
	input_vect, output_vect = s.play(testNN = True, _model)
	
	_input_data.append(input_vect)
	_output_data.append(output_vect)
		
	return _input_data, _output_data


def trainNetworkTF(x, y, model):
	
	model.fit(x,y, n_epoch = 1, shuffle = True, run_id = self.filename)
	model.save(self.filename)

	return model

def create_modelTF():
	
	network = input_data(shape=[None, 7, 1], name='input')
	network = fully_connected(network, 15, activation='relu')
	network = fully_connected(network, 3, activation='softmax')
	network = regression(network, optimizer='adam', learning_rate=1e-2, loss='mean_square', name='target')
	_model = tflearn.DNN(network)
	
	return _model	


def main():
	
	inputData, outputData = getTrainingData(1e2)
	model = create_modelTF()
	trainNetworkTF(model)
	testNetwork(model)

	
if __name__=="__main__":
	main()
	
	

	
