
import matplotlib.pyplot as plt

import pickle

from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.optim import lr_scheduler
# import torchvision
# from torchvision import datasets, models, transforms

# Pennylane
import pennylane as qml
from pennylane import numpy as np

# sklearn
from sklearn.preprocessing import StandardScaler

# Other tools
import time
import os
import copy

# from VQC_GRAD_META_CONSTRUCT import load_JET_4_var_two

from metaquantum.CircuitComponents import *
from metaquantum import Optimization

# Qiskit
import qiskit
import qiskit.providers.aer.noise as noise

# Custom qiskit noise model
from ibm_noise_models import thermal_noise_backend, combined_error_noisy_backend, combined_noise_backend_normdist


# Dataset
# from generate_lstm_dataset import get_sine_data
# from data.load_air_passengers import get_air_passenger_data_single_predict
# from data.damped_shm import get_damped_shm_data
from data.bessel_functions import get_bessel_data
# from data.delayed_quantum_control import get_delayed_quantum_control_data
# from data.population_inversion_revised import get_population_inversion_data
# from generate_lstm_dataset import get_sine_data_single_predict
# from data.narma_data_set import get_narma2_data
# from data.narma_generator import get_narma_data

##
# dtype = torch.cuda.DoubleTensor if torch.cuda.is_available() else torch.DoubleTensor
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

class VQLSTM(nn.Module):
	def __init__(self, 
		lstm_input_size, 
		lstm_hidden_size,
		lstm_output_size,
		lstm_num_qubit,
		lstm_cell_cat_size,
		lstm_cell_num_layers,
		lstm_internal_size,
		duplicate_time_of_input,
		as_reservoir,
		single_y,
		output_all_h,
		qdevice,
		dev,
		gpu_q):

		super().__init__()

		self.lstm_input_size = lstm_input_size
		self.lstm_hidden_size = lstm_hidden_size
		self.lstm_output_size = lstm_output_size
		self.lstm_num_qubit = lstm_num_qubit
		self.lstm_cell_cat_size = lstm_cell_cat_size
		self.lstm_cell_num_layers = lstm_cell_num_layers
		self.lstm_internal_size = lstm_internal_size
		self.duplicate_time_of_input = duplicate_time_of_input
		self.as_reservoir = as_reservoir
		self.single_y = single_y
		self.output_all_h = output_all_h

		self.qdevice = qdevice
		self.dev = dev
		self.gpu_q = gpu_q
		
		
		# self.q_params_1 = nn.Parameter(0.01 * torch.randn(self.rnn_cell_num_layers, self.rnn_num_qubit, 3))
		self.q_params_1 = nn.Parameter(0.01 * torch.randn(self.lstm_cell_num_layers, self.lstm_num_qubit, 3))
		self.q_params_2 = nn.Parameter(0.01 * torch.randn(self.lstm_cell_num_layers, self.lstm_num_qubit, 3))
		self.q_params_3 = nn.Parameter(0.01 * torch.randn(self.lstm_cell_num_layers, self.lstm_num_qubit, 3))
		self.q_params_4 = nn.Parameter(0.01 * torch.randn(self.lstm_cell_num_layers, self.lstm_num_qubit, 3))
		self.q_params_5 = nn.Parameter(0.01 * torch.randn(self.lstm_cell_num_layers, self.lstm_num_qubit, 3))
		self.q_params_6 = nn.Parameter(0.01 * torch.randn(self.lstm_cell_num_layers, self.lstm_num_qubit, 3))

		if self.as_reservoir:
			self.q_params_1.requires_grad = False
			self.q_params_2.requires_grad = False
			self.q_params_3.requires_grad = False
			self.q_params_4.requires_grad = False
			self.q_params_5.requires_grad = False
			self.q_params_6.requires_grad = False
		
		# self.classical_nn_linear = nn.Linear(self.lstm_hidden_size, self.lstm_output_size)
		self.classical_nn_linear = nn.Linear(self.lstm_output_size, 1)


		self.cell_1 = VQCVariationalLoadingFlexNoisy(
			num_of_input= self.lstm_cell_cat_size,
			num_of_output= self.lstm_internal_size,
			num_of_wires = 4,
			num_of_layers = self.lstm_cell_num_layers,
			qdevice = self.qdevice,
			hadamard_gate = True,
			more_entangle = True,
			gpu = self.gpu_q,
			noisy_dev = self.dev)
		self.cell_2 = VQCVariationalLoadingFlexNoisy(
			num_of_input= self.lstm_cell_cat_size,
			num_of_output= self.lstm_internal_size,
			num_of_wires = 4,
			num_of_layers = self.lstm_cell_num_layers,
			qdevice = self.qdevice,
			hadamard_gate = True,
			more_entangle = True,
			gpu = self.gpu_q,
			noisy_dev = dev)
		self.cell_3 = VQCVariationalLoadingFlexNoisy(
			num_of_input= self.lstm_cell_cat_size,
			num_of_output= self.lstm_internal_size,
			num_of_wires = 4,
			num_of_layers = self.lstm_cell_num_layers,
			qdevice = self.qdevice,
			hadamard_gate = True,
			more_entangle = True,
			gpu = self.gpu_q,
			noisy_dev = self.dev)
		self.cell_4 = VQCVariationalLoadingFlexNoisy(
			num_of_input= self.lstm_cell_cat_size,
			num_of_output= self.lstm_internal_size,
			num_of_wires = 4,
			num_of_layers = self.lstm_cell_num_layers,
			qdevice = self.qdevice,
			hadamard_gate = True,
			more_entangle = True,
			gpu = self.gpu_q,
			noisy_dev = self.dev)
		# Transform into h_t
		self.cell_5 = VQCVariationalLoadingFlexNoisy(
			num_of_input= self.lstm_internal_size,
			num_of_output= self.lstm_hidden_size,
			num_of_wires = 4,
			num_of_layers = self.lstm_cell_num_layers,
			qdevice = self.qdevice,
			hadamard_gate = True,
			more_entangle = True,
			gpu = self.gpu_q,
			noisy_dev = self.dev)
		# Transform into output
		self.cell_6 = VQCVariationalLoadingFlexNoisy(
			num_of_input= self.lstm_internal_size,
			num_of_output= self.lstm_output_size,
			num_of_wires = 4,
			num_of_layers = self.lstm_cell_num_layers,
			qdevice = self.qdevice,
			hadamard_gate = True,
			more_entangle = True,
			gpu = self.gpu_q,
			noisy_dev = self.dev)
		

	def get_angles_atan(self, in_x):
		return torch.stack([torch.stack([torch.atan(item), torch.atan(item**2)]) for item in in_x])

	def _forward(self, single_item_x, single_item_h, single_item_c):
		# for the single item input
		self.cell_1.var_Q_circuit = self.q_params_1
		self.cell_2.var_Q_circuit = self.q_params_2
		self.cell_3.var_Q_circuit = self.q_params_3
		self.cell_4.var_Q_circuit = self.q_params_4
		self.cell_5.var_Q_circuit = self.q_params_5
		self.cell_6.var_Q_circuit = self.q_params_6

		single_item_x = torch.cat([single_item_x for i in range(self.duplicate_time_of_input)])
		cat = torch.cat([single_item_x, single_item_h])
		# print("cat: ", cat)

		res_temp = self.get_angles_atan(cat)
		# print("res_temp: ", res_temp)

		res_from_cell_1 = self.cell_1.forward(res_temp)
		act_1 = nn.Sigmoid()
		res_from_cell_1 = act_1(res_from_cell_1)
		# print("res_from_cell_1: ", res_from_cell_1)
		res_from_cell_2 = self.cell_2.forward(res_temp)
		act_2 = nn.Sigmoid()
		res_from_cell_2 = act_2(res_from_cell_2)
		# print("res_from_cell_2: ", res_from_cell_2)
		res_from_cell_3 = self.cell_3.forward(res_temp)
		act_3 = nn.Tanh()
		res_from_cell_3 = act_3(res_from_cell_3)
		# print("res_from_cell_3: ", res_from_cell_3)
		res_from_cell_4 = self.cell_4.forward(res_temp)
		act_4 = nn.Sigmoid()
		res_from_cell_4 = act_4(res_from_cell_4)
		# print("res_from_cell_4: ", res_from_cell_4)


		res_2_mul_3 = torch.mul(res_from_cell_2, res_from_cell_3)
		res_c = torch.mul(single_item_c, res_from_cell_1)
		c_t = torch.add(res_c, res_2_mul_3) 
		h_t = self.cell_5.forward(self.get_angles_atan(torch.mul(res_from_cell_4, torch.tanh(c_t)))) 
		# out = self.classical_final_scaling[0] * self.cell_6.forward(self.get_angles_atan(torch.mul(res_from_cell_4, torch.tanh(c_t)))) + self.classical_final_scaling[1]
		
		cell_6_res = self.cell_6.forward(self.get_angles_atan(torch.mul(res_from_cell_4, torch.tanh(c_t))))
		# out = self.classical_final_scaling[0] * self.cell_6.forward(self.get_angles_atan(torch.mul(res_from_cell_4, torch.tanh(c_t)))) + self.classical_final_scaling[1]
		# print(cell_6_res)
		out = self.classical_nn_linear.forward(cell_6_res)

		return h_t, c_t, out

	def forward(self, input_sequence_x, initial_h, initial_c):


		h = initial_h.clone().detach()
		c = initial_c.clone().detach()

		seq = []

		h_history = []
		c_history = [] 

		for item in input_sequence_x:
			h, c, out = self._forward(item, h, c)
			h_history.append(h)
			c_history.append(c)
			# print(h)
			# print(c)
			# print(out)

			seq.append(out)

		if self.output_all_h:
			return torch.stack(h_history), torch.stack(c_history)
		else:

			return seq[-1]





def main():

	dtype = torch.DoubleTensor
	device = 'cpu'


	qdevice = "default.qubit" 
	# qdevice = "qulacs.simulator"

	# gpu_q = True
	gpu_q = False

	##
	duplicate_time_of_input = 1

	lstm_input_size = 1
	lstm_hidden_size = 3
	lstm_cell_cat_size = lstm_input_size + lstm_hidden_size
	lstm_internal_size = 4
	lstm_output_size = 4  
	lstm_cell_num_layers = 4 
	lstm_num_qubit = 4

	as_reservoir = False


	use_qiskit_noise_model = False

	dev = None

	if use_qiskit_noise_model:
		noise_model = combined_noise_backend_normdist(num_qubits = lstm_num_qubit)
		dev = qml.device('qiskit.aer', wires=lstm_num_qubit, noise_model=noise_model)

	else:
		dev = qml.device("default.qubit", wires = lstm_num_qubit)


	model = VQLSTM(lstm_input_size = lstm_input_size, 
		lstm_hidden_size = lstm_hidden_size,
		lstm_output_size = lstm_output_size,
		lstm_num_qubit = lstm_num_qubit,
		lstm_cell_cat_size = lstm_cell_cat_size,
		lstm_cell_num_layers = lstm_cell_num_layers,
		lstm_internal_size = lstm_internal_size,
		duplicate_time_of_input = duplicate_time_of_input,
		as_reservoir = as_reservoir,
		single_y = False,
		output_all_h = True,
		qdevice = qdevice,
		dev = dev,
		gpu_q = gpu_q).double()

	# Load the data

	x, y = get_bessel_data()

	num_for_train_set = int(0.67 * len(x))

	x_train = x[:num_for_train_set].type(dtype)
	y_train = y[:num_for_train_set].type(dtype)

	x_test = x[num_for_train_set:].type(dtype)
	y_test = y[num_for_train_set:].type(dtype)

	print("x_train: ", x_train)
	print("x_test: ", x_test)


	h_0 = torch.zeros(lstm_hidden_size,).type(dtype)
	c_0 = torch.zeros(lstm_internal_size,).type(dtype)

	print("First data: ", x_train[0])
	print("First target: ", y_train[0])
	first_run_h, first_run_c = model.forward(x_train[0].reshape(4,1), h_0, c_0)
	print("Output of first_run_h: ", first_run_h)
	print("Output of first_run_c: ", first_run_c)

	lstm_input_size = 3
	lstm_hidden_size = 1
	lstm_cell_cat_size = lstm_input_size + lstm_hidden_size
	lstm_internal_size = 4
	lstm_output_size = 4
	lstm_cell_num_layers = 4
	lstm_num_qubit = 4

	model_2 = VQLSTM(lstm_input_size = lstm_input_size, 
		lstm_hidden_size = lstm_hidden_size,
		lstm_output_size = lstm_output_size,
		lstm_num_qubit = lstm_num_qubit,
		lstm_cell_cat_size = lstm_cell_cat_size,
		lstm_cell_num_layers = lstm_cell_num_layers,
		lstm_internal_size = lstm_internal_size,
		duplicate_time_of_input = duplicate_time_of_input,
		as_reservoir = as_reservoir,
		single_y = False,
		output_all_h = True,
		qdevice = qdevice,
		dev = dev,
		gpu_q = gpu_q).double()

	test_input = torch.zeros(4,3).type(dtype)
	first_run_h, first_run_c = model_2.forward(test_input, h_0, c_0)
	print("Output of model_2 first_run_h: ", first_run_h)
	print("Output Shape of model_2 first_run_h: ", first_run_h.shape)

	print("Output of model_2 first_run_c: ", first_run_c)
	print("Output Shape of model_2 first_run_c: ", first_run_c.shape)

	res_from_model_1_h, res_from_model_1_c = model.forward(x_train[0].reshape(4,1), h_0, c_0)
	print("Output of MODEL 1 h: ", res_from_model_1_h)
	print("Output Shape of MODEL 1 h: ", res_from_model_1_h.shape)

	print("Output of MODEL 1 c: ", res_from_model_1_c)
	print("Output Shape of MODEL 1 c: ", res_from_model_1_c.shape)

	second_run_h, second_run_c = model_2.forward(res_from_model_1_h, h_0, c_0)
	print("Output of (SENDING H FROM MODEL 1 INTO MODEL 2) h: ", second_run_h)
	print("Output shape of (SENDING H FROM MODEL 1 INTO MODEL 2) h: ", second_run_h.shape)

	print("Output of (SENDING H FROM MODEL 1 INTO MODEL 2) c: ", second_run_c)
	print("Output shape of (SENDING H FROM MODEL 1 INTO MODEL 2) c: ", second_run_c.shape)



	return


if __name__ == '__main__':
	main()