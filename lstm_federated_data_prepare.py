import torch


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



from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataSet(Dataset):
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def __len__(self):
		return len(self.x)


	def __getitem__(self, index):
		return self.x[index], self.y[index]



def main():
	x, y = get_bessel_data()

	print("x shape: ", x.shape)
	print("y shape: ", y.shape)
	data = TimeSeriesDataSet(x, y)

	print(len(data))

	# Test the dataloader
	time_series_data_loader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=True)

	for data, target in time_series_data_loader:
		print(data.shape)
		print(target.shape)

	return


if __name__ == '__main__':
	main()