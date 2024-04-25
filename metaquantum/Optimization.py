import os

import pennylane as qml
from pennylane import numpy as np
# import numpy as np
# from pennylane.optimize import NesterovMomentumOptimizer

import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torch.nn as nn 
from torch.autograd import Variable
import torch.multiprocessing as mp

from . import Utils

import time




def accuracy(labels, predictions):
	""" Share of equal labels and predictions

	Args:
		labels (array[float]): 1-d array of labels
		predictions (array[float]): 1-d array of predictions
	Returns:
		float: accuracy
	"""

	loss = 0
	for l, p in zip(labels, predictions):
		if abs(l.item() - p) < 1e-2:
			loss = loss + 1
	loss = loss / len(labels)


	return loss


def lost_function_cross_entropy(labels, predictions):
	loss = nn.CrossEntropyLoss()
	output = loss(predictions, labels)
	print("LOSS AVG: ",output)
	return output



def cost(VQC, X, Y):
	"""Cost (error) function to be minimized."""

	# predictions = torch.stack([variational_classifier(var_Q_circuit = var_Q_circuit, var_Q_bias = var_Q_bias, angles=item) for item in X])

	loss = nn.CrossEntropyLoss()
	output = loss(torch.stack([VQC.forward(item) for item in X]), Y)
	print("LOSS AVG: ",output)
	return output

def MSEcost(VQC, X, Y):
	"""Cost (error) function to be minimized."""

	# predictions = torch.stack([variational_classifier(var_Q_circuit = var_Q_circuit, var_Q_bias = var_Q_bias, angles=item) for item in X])

	loss = nn.MSELoss()
	output = loss(torch.stack([VQC.forward(item) for item in X]), Y)
	print("LOSS AVG: ",output)
	return output


def cost_function(VQC, LossFunction, X, Y):
	"""Cost (error) function to be minimized."""

	# predictions = torch.stack([variational_classifier(var_Q_circuit = var_Q_circuit, var_Q_bias = var_Q_bias, angles=item) for item in X])
	
	output = LossFunction(torch.stack([VQC.forward(item) for item in X]), Y)
	print("LOSS AVG: ",output)
	return output


def train_epoch_full(opt, VQC, X, Y, batch_size):
	losses = []
	for beg_i in range(0, X.shape[0], batch_size):
		X_train_batch = X[beg_i:beg_i + batch_size]
		# print(x_batch.shape)
		Y_train_batch = Y[beg_i:beg_i + batch_size]

		# opt.step(closure)
		since_batch = time.time()
		opt.zero_grad()
		print("CALCULATING LOSS...")
		loss = cost(VQC = VQC, X = X_train_batch, Y = Y_train_batch)
		print("BACKWARD..")
		loss.backward()
		losses.append(loss.data.cpu().numpy())
		opt.step()
# 		print("LOSS IN CLOSURE: ", loss)
		print("FINISHED OPT.")
		print("Batch time: ", time.time() - since_batch)
		# print("CALCULATING PREDICTION.")
	losses = np.array(losses)
	return losses.mean()

def train_epoch(opt, VQC, X, Y, batch_size, sampling_iteration):
	""" train epoch, each epoch is 100 times random sampling"""
	losses = []
	for i in range(sampling_iteration):

		# Test Saving
		Utils.Saving.save_test()
		since_batch = time.time()


		batch_index = np.random.randint(0, len(X), (batch_size, ))
		X_train_batch = X[batch_index]
		Y_train_batch = Y[batch_index]
		# opt.step(closure)
		opt.zero_grad()
		print("CALCULATING LOSS...")
		loss = cost(VQC = VQC, X = X_train_batch, Y = Y_train_batch)
		print("BACKWARD..")
		loss.backward()
		losses.append(loss.data.cpu().numpy())
		opt.step()
	# 		print("LOSS IN CLOSURE: ", loss)
		print("FINISHED OPT.")
		print("Batch time: ", time.time() - since_batch)
		# print("CALCULATING PREDICTION.")
	losses = np.array(losses)
	return losses.mean()

def BinaryCrossEntropy(opt, vqc, X, Y, batch_size):

	return train_epoch(opt, vqc, X, Y, batch_size)


def train_model(opt, 
	VQC, 
	x_for_train, 
	y_for_train, 
	x_for_val, 
	y_for_val, 
	x_for_test, 
	y_for_test,
	exp_name,
	exp_index,
	saving_files = True, 
	batch_size = 10, 
	epoch_num = 100, 
	sampling_iteration = 100, 
	full_epoch = False,
	show_params = False,
	torch_first_model = False):

	iter_index = []
	cost_train_list = []
	cost_test_list = []
	acc_train_list = []
	acc_val_list = []
	acc_test_list = []

	file_title = exp_name + datetime.now().strftime("NO%Y%m%d%H%M%S")
	exp_name = exp_name + "Exp_" + str(exp_index)


	var_Q_circuit = ''
	if torch_first_model == True:
		var_Q_circuit = VQC.state_dict()
	else:
		var_Q_circuit = VQC.var_Q_array

	var_Q_bias = ''

	# print(var_Q_circuit)

	if not os.path.exists(exp_name):
		os.makedirs(exp_name)

	if saving_files == True:
		Utils.Saving.save_training_and_testing(exp_name = exp_name, file_title = file_title, training_x = x_for_train, training_y = y_for_train, val_x = x_for_val, val_y = y_for_val, testing_x = x_for_test, testing_y = y_for_test)
	# print(VQC.var_Q_array)
	if show_params == True:
		print(var_Q_circuit)
	for it in range(epoch_num):
		if full_epoch == True:
			avg_loss_in_epoch = train_epoch_full(opt, VQC, x_for_train, y_for_train, batch_size)

		else:
			avg_loss_in_epoch = train_epoch(opt, VQC, x_for_train, y_for_train, batch_size, sampling_iteration)
		if show_params == True:
			print(var_Q_circuit)

		# print(var_Q_circuit)
		# print(var_Q_circuit_1)
		# print(var_Q_bias_1)
		# print(var_Q_circuit_2)
		# print(var_Q_bias_2)
		# print(var_Q_circuit_3)
		# print(var_Q_bias_3)

		# print(VQC.var_Q_array)

		
		print("CALCULATE PRED TRAIN ... ")
		predictions_train = [torch.argmax(VQC.forward(item)).item() for item in x_for_train]
		print("CALCULATE PRED VALIDATION ... ")
		predictions_val = [torch.argmax(VQC.forward(item)).item() for item in x_for_val]
		print("CALCULATE PRED TEST ... ")
		predictions_test = [torch.argmax(VQC.forward(item)).item() for item in x_for_test]
		print("CALCULATE TRAIN COST ... ")

	
		# cost_train = cost_for_result(VQC, x_train, y_train).item()
		cost_train = cost(VQC, x_for_train, y_for_train).item()

		print("CALCULATE TEST COST ... ")
		# cost_test = cost_for_result(VQC, x_test, y_test).item()
		cost_test = cost(VQC, x_for_test, y_for_test).item()

		# print('Y_for_train: ',Y_for_train)
		# print('predictions_train: ', predictions_train)

		acc_train = accuracy(y_for_train, predictions_train)
		acc_val = accuracy(y_for_val, predictions_val)
		acc_test = accuracy(y_for_test, predictions_test)

		iter_index.append(it+1)
		acc_train_list.append(acc_train)
		acc_val_list.append(acc_val)
		acc_test_list.append(acc_test)
		cost_train_list.append(cost_train)
		cost_test_list.append(cost_test)


		if saving_files == True:
			Utils.Saving.save_all_the_current_info(exp_name, file_title, iter_index, var_Q_circuit, var_Q_bias, cost_train_list, cost_test_list, acc_train_list, acc_val_list, acc_test_list)
			if torch_first_model == True:
				Utils.Saving.saving_torch_model(exp_name, file_title, iter_index, VQC.state_dict())



		print("Epoch: {:5d} | Cost train: {:0.7f} | Cost test: {:0.7f} | Acc train: {:0.7f} | Acc validation: {:0.7f} | Acc test: {:0.7f}"
			  "".format(it+1, cost_train, cost_test, acc_train, acc_val, acc_test))

	return