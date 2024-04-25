import matplotlib.pyplot as plt

# Saving
import pickle

# Datetime
from datetime import datetime

# PyTorch
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
# Device auto select
# dtype = torch.cuda.DoubleTensor if torch.cuda.is_available() else torch.DoubleTensor
# device = 'cuda' if torch.cuda.is_available() else 'cpu'


from lstm_base_class import VQLSTM
from lstm_federated_data_prepare import TimeSeriesDataSet


## Training
def MSEcost(VQC, X, Y, h_0, c_0, seq_len):
	"""Cost (error) function to be minimized."""

	# predictions = torch.stack([variational_classifier(var_Q_circuit = var_Q_circuit, var_Q_bias = var_Q_bias, angles=item) for item in X])

	loss = nn.MSELoss()
	output = loss(torch.stack([VQC.forward(vec.reshape(seq_len,1), h_0, c_0).reshape(1,) for vec in X]), Y.reshape(Y.shape[0],1))
	print("LOSS AVG: ",output)
	return output

def train_epoch_full(opt, VQC, data, h_0, c_0, seq_len, batch_size):
	losses = []
	time_series_data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

	for X_train_batch, Y_train_batch in time_series_data_loader:

		since_batch = time.time()
		opt.zero_grad()
		print("CALCULATING LOSS...")
		loss = MSEcost(VQC = VQC, X = X_train_batch, Y = Y_train_batch, h_0 = h_0, c_0 = c_0, seq_len = seq_len)
		print("BACKWARD..")
		loss.backward()
		losses.append(loss.data.cpu().numpy())
		opt.step()
# 		print("LOSS IN CLOSURE: ", loss)
		print("FINISHED OPT.")
		print("Batch time: ", time.time() - since_batch)

	losses = np.array(losses)
	return losses.mean()


def saving(exp_name, exp_index, train_len, iteration_list, train_loss_list, test_loss_list, model, simulation_result, ground_truth):
	file_name = exp_name + "_NO_" + str(exp_index) + "_Epoch_" + str(iteration_list[-1])
	saved_simulation_truth = {
	"simulation_result" : simulation_result,
	"ground_truth" : ground_truth
	}

	if not os.path.exists(exp_name):
		os.makedirs(exp_name)

	# Save the train loss list
	with open(exp_name + "/" + file_name + "_TRAINING_LOST" + ".txt", "wb") as fp:
		pickle.dump(train_loss_list, fp)

	# Save the test loss list
	with open(exp_name + "/" + file_name + "_TESTING_LOST" + ".txt", "wb") as fp:
		pickle.dump(test_loss_list, fp)

	# Save the simulation result
	with open(exp_name + "/" + file_name + "_SIMULATION_RESULT" + ".txt", "wb") as fp:
		pickle.dump(saved_simulation_truth, fp)

	# Save the model parameters
	torch.save(model.state_dict(), exp_name + "/" +  file_name + "_torch_model.pth")

	# Plot
	plotting_data(exp_name, exp_index, file_name, iteration_list, train_loss_list, test_loss_list)
	plotting_simulation(exp_name, exp_index, file_name, train_len, simulation_result, ground_truth)

	return


def plotting_data(exp_name, exp_index, file_name, iteration_list, train_loss_list, test_loss_list):
	# Plot train and test loss
	fig, ax = plt.subplots()
	# plt.yscale('log')
	ax.plot(iteration_list, train_loss_list, '-b', label='Training Loss')
	ax.plot(iteration_list, test_loss_list, '-r', label='Testing Loss')
	leg = ax.legend();

	ax.set(xlabel='Epoch', 
		   title=exp_name)
	fig.savefig(exp_name + "/" + file_name + "_" + "loss" + "_"+ datetime.now().strftime("NO%Y%m%d%H%M%S") + ".pdf", format='pdf')
	plt.clf()

	return

def plotting_simulation(exp_name, exp_index, file_name, train_len, simulation_result, ground_truth):
	# Plot the simulation
	plt.axvline(x=train_len, c='r', linestyle='--')
	plt.plot(simulation_result, '-')
	plt.plot(ground_truth.detach().numpy(), '--')
	plt.suptitle(exp_name)
	# savfig can only be placed BEFORE show()
	plt.savefig(exp_name + "/" + file_name + "_" + "simulation" + "_"+ datetime.now().strftime("NO%Y%m%d%H%M%S") + ".pdf", format='pdf')
	return

#



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
	lstm_cell_num_layers = 2 
	lstm_num_qubit = 4

	as_reservoir = False

	use_qiskit_noise_model = False


	dev = None

	if use_qiskit_noise_model:
		noise_model = combined_noise_backend_normdist(num_qubits = lstm_num_qubit)
		dev = qml.device('qiskit.aer', wires=lstm_num_qubit, noise_model=noise_model)

	else:
		dev = qml.device("default.qubit", wires = lstm_num_qubit)


	# Initialize the model
	model = VQLSTM(lstm_input_size = lstm_input_size, 
		lstm_hidden_size = lstm_hidden_size,
		lstm_output_size = lstm_output_size,
		lstm_num_qubit = lstm_num_qubit,
		lstm_cell_cat_size = lstm_cell_cat_size,
		lstm_cell_num_layers = lstm_cell_num_layers,
		lstm_internal_size = lstm_internal_size,
		duplicate_time_of_input = duplicate_time_of_input,
		as_reservoir = as_reservoir,
		single_y = True,
		output_all_h = False,
		qdevice = qdevice,
		dev = dev,
		gpu_q = gpu_q).double()

	# Load the data

	x, y = get_bessel_data()

	num_for_train_set = int(0.67 * len(x))

	x_train = x[:num_for_train_set].type(dtype)
	y_train = y[:num_for_train_set].type(dtype)

	train_data = TimeSeriesDataSet(x_train, y_train)

	x_test = x[num_for_train_set:].type(dtype)
	y_test = y[num_for_train_set:].type(dtype)

	test_data = TimeSeriesDataSet(x_test, y_test)

	print("x_train: ", x_train)
	print("x_test: ", x_test)


	h_0 = torch.zeros(lstm_hidden_size,).type(dtype)
	c_0 = torch.zeros(lstm_internal_size,).type(dtype)

	print("First data: ", x_train[0])
	print("First target: ", y_train[0])
	# first_run_h, first_run_c = model.forward(x_train[0].reshape(4,1), h_0, c_0)
	first_run = model.forward(x_train[0].reshape(4,1), h_0, c_0)
	# print("Output of first_run_h: ", first_run_h)
	# print("Output of first_run_c: ", first_run_c)


	exp_name = "VQ_LSTM_BASE_CLASS_TS_MODEL__DATALOADER_BESSEL_{}_QUBIT".format(lstm_num_qubit)


	if as_reservoir:
		exp_name += "_AS_RESERVOIR"

	if use_qiskit_noise_model:
		exp_name += "_QISKIT_NOISE"


	exp_name += "_{}_QuLAYERS".format(lstm_cell_num_layers)

	exp_index = 2
	train_len = len(x_train)

	opt = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
	train_loss_for_all_epoch = []
	test_loss_for_all_epoch = []
	iteration_list = []
	for i in range(100):
		iteration_list.append(i + 1)
		c_0 = torch.zeros(lstm_internal_size,).type(dtype)
		h_0 = torch.zeros(lstm_hidden_size,).type(dtype)


		train_loss_epoch = train_epoch_full(opt = opt, VQC = model, data = train_data, h_0 = h_0, c_0 = c_0, seq_len = 4,batch_size = 10)
		print("c_0: ", c_0)
		print("h_0: ", h_0)
		test_loss = MSEcost(VQC = model, X = x_test, Y = y_test, h_0 = h_0, c_0 = c_0, seq_len = 4)
		print("TEST LOSS: ", test_loss)

		# train_loss_for_all_epoch.append(train_loss_epoch)
		# test_loss_for_all_epoch.append(test_loss)
		train_loss_for_all_epoch.append(train_loss_epoch.numpy())
		test_loss_for_all_epoch.append(test_loss.detach().numpy())

		plot_each_epoch = True
		if plot_each_epoch == True:
			total_res = None
			ground_truth_y = None
			if device == 'cuda':
				total_res = torch.stack([model.forward(vec.reshape(4,1), h_0, c_0).reshape(1,) for vec in x.type(dtype)]).detach().cpu().numpy()
				ground_truth_y = y.clone().detach().cpu()
			else:
				total_res = torch.stack([model.forward(vec.reshape(4,1), h_0, c_0).reshape(1,) for vec in x.type(dtype)]).detach().numpy()
				ground_truth_y = y.clone().detach()

			saving(
				exp_name = exp_name, 
				exp_index = exp_index, 
				train_len = train_len, 
				iteration_list = iteration_list, 
				train_loss_list = train_loss_for_all_epoch, 
				test_loss_list = test_loss_for_all_epoch, 
				model = model, 
				simulation_result = total_res, 
				ground_truth = ground_truth_y)

	return




if __name__ == '__main__':
	main()