# updated Bessel function datasets with different number of points

import numpy as np
import scipy.special
import matplotlib.pyplot as plt

import torch
import torch.nn as nn 
from torch.autograd import Variable
import torch.optim as optim

from sklearn.preprocessing import MinMaxScaler

# create a figure window
fig = plt.figure(1, figsize=(9,8))


def generate_dataset(data_src = None):
	# normalize the dataset
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaled_dataset = scaler.fit_transform(data_src.reshape(-1, 1))
	return scaled_dataset 

def transform_data_single_predict(data, seq_length):
	x = []
	y = []

	for i in range(len(data)-seq_length-1):
		_x = data[i:(i+seq_length)]
		_y = data[i+seq_length]
		x.append(_x)
		y.append(_y)
	x_var = Variable(torch.from_numpy(np.array(x).reshape(-1, seq_length)).float())
	y_var = Variable(torch.from_numpy(np.array(y)).float())

	return x_var, y_var

def get_bessel_data(data = "j2", num_points = 1000, seq_len = 4):
	# create arrays for a few Bessel functions and plot them
	x = np.linspace(2, 100, num_points)
	j0 = scipy.special.jn(0, x)
	j1 = scipy.special.jn(1, x)
	j2 = scipy.special.jn(2, x)
	y0 = scipy.special.yn(0, x)
	y1 = scipy.special.yn(1, x)
	y2 = scipy.special.yn(2, x)

	bessel_data = None

	if data == "j0":
		bessel_data = j0
	elif data == "j1":
		bessel_data = j1
	elif data == "j2":
		bessel_data = j2
	elif data == "y0":
		bessel_data = y0
	elif data == "y1":
		bessel_data = y1
	elif data == "y2":
		bessel_data = y2

	scaled_dataset = generate_dataset(bessel_data)
	return transform_data_single_predict(data = scaled_dataset, seq_length = seq_len)

def plotting_test(data_src):
	ax1 = fig.add_subplot()
	ax1.plot(x,data_src)
	ax1.axhline(color="grey", ls="--", zorder=-1)
	ax1.set_ylim(-1,1)
	ax1.text(0.5, 0.95,'Bessel', ha='center', va='top',
		 transform = ax1.transAxes)

	plt.show()

def main():
	# scaled_dataset = generate_dataset(j2)
	# plotting_test(scaled_dataset)
	x, y = get_bessel_data(data = "j2", num_points = 3000, seq_len = 4)
	print(x.shape)
	print(y.shape)
	num_for_train_set = int(0.67 * len(x))
	print(num_for_train_set)

	

if __name__ == '__main__':
	main()
