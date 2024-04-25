import os
import random
from tqdm import tqdm
import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.utils.data.dataset import Dataset  

import pickle
import matplotlib.pyplot as plt
from datetime import datetime

dtype = torch.cuda.DoubleTensor if torch.cuda.is_available() else torch.DoubleTensor
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from lstm_base_class import VQLSTM
from lstm_federated_data_prepare import TimeSeriesDataSet

# Pennylane
import pennylane as qml

# Dataset
# from generate_lstm_dataset import get_sine_data
# from data.load_air_passengers import get_air_passenger_data_single_predict
# from data.damped_shm import get_damped_shm_data
from data.bessel_functions_new import get_bessel_data
# from data.delayed_quantum_control import get_delayed_quantum_control_data
# from data.population_inversion_revised import get_population_inversion_data
# from generate_lstm_dataset import get_sine_data_single_predict


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

def simulation_test(VQC, X, Y, h_0, c_0):

	total_res = torch.stack([VQC.forward(vec.reshape(4,1), h_0, c_0).reshape(1,) for vec in X.type(dtype)]).detach().numpy()
	ground_truth_y = Y.clone().detach()

	return total_res, ground_truth_y


num_clients = 100
num_selected = 10 
num_rounds = 100
epochs = 1   
batch_size = 4


x, y = get_bessel_data(data = "j2", num_points = 3000, seq_len = 4)

num_for_train_set = int(0.67 * len(x))

x_train = x[:num_for_train_set].type(dtype)
y_train = y[:num_for_train_set].type(dtype)

train_len = len(x_train)

x_train = x_train[:2000]
y_train = y_train[:2000]
print("NUM TRAIN: ", len(x_train))

train_data = TimeSeriesDataSet(x_train, y_train)

train_data_split = torch.utils.data.random_split(train_data, [int(len(x_train) / num_clients) for _ in range(num_clients)])

train_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True) for x in train_data_split]

x_test = x[num_for_train_set:].type(dtype)
y_test = y[num_for_train_set:].type(dtype)

test_data = TimeSeriesDataSet(x_test, y_test)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)



def saving(exp_name, exp_index, train_len, iteration_list, train_loss_list, test_loss_list, model, simulation_result, ground_truth):
	# Generate file name
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


# Initialize the model
qdevice = "default.qubit" 
gpu_q = False
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





def client_update(client_model, optimizer, train_loader, epoch=5):
	"""
	This function updates/trains client model on client data
	"""
	client_model.train()
	for e in range(epoch):
		print("EPOCH: ",e)
		c_0 = torch.zeros(lstm_internal_size,).type(dtype)
		h_0 = torch.zeros(lstm_hidden_size,).type(dtype)

		for batch_idx, (data, target) in enumerate(train_loader):
			print("BATCH IDX: ", batch_idx)
			# data, target = data.to(device), target.to(device)
			# optimizer.zero_grad()
			# output = client_model(data)
			# criterion = nn.CrossEntropyLoss()

			# loss = F.nll_loss(output, target)
			# loss = criterion(output, target)
			loss = MSEcost(VQC = client_model, X = data, Y = target, h_0 = h_0, c_0 = c_0, seq_len = 4)
			print("Loss: ", loss)
			loss.backward()
			optimizer.step()
	return loss.item()


def server_aggregate(global_model, client_models):
	"""
	This function has aggregation method 'mean'
	"""
	global_dict = global_model.state_dict()
	for k in global_dict.keys():
		global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0)
	global_model.load_state_dict(global_dict)
	for model in client_models:
		model.load_state_dict(global_model.state_dict())





def test(global_model, X, Y):
	"""This function test the global model on test data and returns test loss and test accuracy """
	global_model.eval()
	c_0 = torch.zeros(lstm_internal_size,).type(dtype)
	h_0 = torch.zeros(lstm_hidden_size,).type(dtype)
	test_loss = MSEcost(VQC = global_model, X = X, Y = Y, h_0 = h_0, c_0 = c_0, seq_len = 4)

	return test_loss


global_model = VQLSTM(lstm_input_size = lstm_input_size, 
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


client_models = []

for i in range(num_selected):
	client_qlstm = VQLSTM(lstm_input_size = lstm_input_size, 
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
	client_models.append(client_qlstm)



for model in client_models:
	model.load_state_dict(global_model.state_dict()) 
# opt = [optim.SGD(model.parameters(), lr=0.1) for model in client_models]

# torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
opt = [torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False) for model in client_models]




exp_index = 2
folder_name = "FEDERATED_QLSTM_TS_MODEL_BESSEL_EXP_{}".format(exp_index)
# exp_name = "FEDERATED_QLSTM_TS_MODEL_BESSEL"

if not os.path.exists(folder_name):
	os.makedirs(folder_name)


losses_train = []
losses_test = []
# acc_train = []
# acc_test = []
epoch_list = []
# Runnining FL

for r in range(num_rounds):
	epoch_list.append(r + 1)
	# select random clients
	client_idx = np.random.permutation(num_clients)[:num_selected]
	# client update
	loss = 0
	for i in tqdm(range(num_selected)):
		loss += client_update(client_models[i], opt[i], train_loader[client_idx[i]], epoch=epochs)
	
	losses_train.append((loss / num_selected))
	# server aggregate
	server_aggregate(global_model, client_models)

	# Need to test the global model via simulation and plot

	c_0 = torch.zeros(lstm_internal_size,).type(dtype)
	h_0 = torch.zeros(lstm_hidden_size,).type(dtype)
	
	total_res, ground_truth_y = simulation_test(VQC = global_model, X = x, Y = y, h_0 = h_0, c_0 = c_0)
	
	test_loss = test(global_model, x_test, y_test)
	losses_test.append(test_loss.detach().numpy())
	# acc_test.append(acc)
	print('%d-th round' % r)
	print('average train loss %0.3g | test loss %0.3g ' % (loss / num_selected, test_loss))
	


	saving(
		exp_name = folder_name, 
		exp_index = exp_index, 
		train_len = train_len, 
		iteration_list = epoch_list, 
		train_loss_list = losses_train, 
		test_loss_list = losses_test, 
		model = global_model, 
		simulation_result = total_res, 
		ground_truth = ground_truth_y)




















