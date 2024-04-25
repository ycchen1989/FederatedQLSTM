from timeit import default_timer as timer

# import numpy as np
from pennylane import numpy as np
import warnings
import torch
import torch.nn as nn 
from torch.autograd import Variable

import pennylane as qml

import torch.multiprocessing as mp


# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import make_pipeline


dtype = torch.cuda.DoubleTensor if torch.cuda.is_available() else torch.DoubleTensor
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class VQCBaseClass:
	def __init__(
			self,
			num_of_input= 10,
			num_of_output= 4,
			num_of_wires = 10,
			num_of_layers = 2,
			var_Q_circuit = None,
			var_Q_bias = None,
			qdevice = "default.qubit",
			gpu = False):

		self.var_Q_circuit = var_Q_circuit
		self.var_Q_bias = var_Q_bias
		self.num_of_input = num_of_input
		self.num_of_output = num_of_output
		self.num_of_wires = num_of_wires
		self.num_of_layers = num_of_layers
		self.qdevice = qdevice


		if gpu == True and qdevice == "qulacs.simulator":
			self.dev = qml.device(self.qdevice, wires = num_of_wires, gpu = True)
		else:
			self.dev = qml.device(self.qdevice, wires = num_of_wires)

	def set_params(self, var_Q_circuit, var_Q_bias):
		self.var_Q_circuit = var_Q_circuit
		self.var_Q_bias = var_Q_bias

	def init_params(self):
		self.var_Q_circuit = Variable(torch.tensor(0.01 * np.random.randn(self.num_of_layers, self.num_of_wires, 3), device=device).type(dtype), requires_grad=True)
		return self.var_Q_circuit

	def _statepreparation(self, angles):
		return angles

	def _finalprocessing(self,vec):
		return vec

	def _angletransformation(self, angles):
		return angles

	def _layer(self, W):
		""" Single layer of the variational classifier.

		Args:
			W (array[float]): 2-d array of variables for one layer

		"""

		# W = W.numpy()

		# Entanglement Layer

		for i in range(self.num_of_wires):
			qml.CNOT(wires=[i, (i + 1) % self.num_of_wires])

		# Rotation Layer
		for j in range(self.num_of_wires):
			qml.Rot(W[j, 0], W[j, 1], W[j, 2], wires=j)

	def circuit(self, angles):

		@qml.qnode(self.dev, interface='torch')
		def _circuit(var_Q_circuit, angles):
			"""The circuit of the variational classifier."""

			# State Preparation
			# qml.QubitStateVector(angles, wires=list(range(self.num_of_wires)))

			self._statepreparation(angles)

			weights = var_Q_circuit
			
			for W in weights:
				self._layer(W)


			return [qml.expval(qml.PauliZ(k)) for k in range(self.num_of_output)]

		return _circuit(self.var_Q_circuit, angles)

	def _forward(self, angles):
		"""The variational classifier."""
		# weights = self.var_Q_circuit
		bias = self.var_Q_bias 

		# angles = angles / torch.sqrt(torch.sum(angles ** 2))

		# angle transformation

		# angles = angles / torch.clamp(torch.sqrt(torch.sum(angles ** 2)), min = 1e-9)
		angles = self._angletransformation(angles)

		# print("ANGLES: ", angles)

		# print("ANGLES in VC:")
		# print(angles)

		# raw_output = torch.tensor(circuit(weights, angles=angles), dtype = torch.double)
		raw_output = self.circuit(angles)

		output = self._finalprocessing(raw_output)

		return output

	def forward(self, angles):
		# Output torch tensor

		# angles = angles[0]
		# angles = torch.from_numpy(angles)
		# fw = np.array([self._forward(angles).detach().numpy()])
		# fw = self._forward(angles).detach()
		fw = self._forward(angles)
		# print("FW:",fw)
		return fw


class VQCVariationalLoading(VQCBaseClass):
	def __init__(
			self,
			num_of_input= 10,
			num_of_output= 4,
			num_of_wires = 10,
			num_of_layers = 2,
			var_Q_circuit = None,
			var_Q_bias = None,
			qdevice = "default.qubit",
			gpu = False):
		super().__init__(num_of_input, num_of_output, num_of_wires, num_of_layers, var_Q_circuit, var_Q_bias, qdevice, gpu)

	def _statepreparation(self, angles):
		for i in range(self.num_of_input):
			qml.RY(angles[i,0], wires=i)
			qml.RZ(angles[i,1], wires=i)

class VQCVariationalLoadingFlex(VQCBaseClass):
	def __init__(
			self,
			num_of_input= 10,
			num_of_output= 4,
			num_of_wires = 10,
			num_of_layers = 2,
			var_Q_circuit = None,
			var_Q_bias = None,
			qdevice = "default.qubit",
			hadamard_gate = False,
			more_entangle = False,
			gpu = False):
		super().__init__(num_of_input, num_of_output, num_of_wires, num_of_layers, var_Q_circuit, var_Q_bias, qdevice, gpu)
		self.hadamard_gate = hadamard_gate
		self.more_entangle = more_entangle

	def _layer(self, W):
		""" Single layer of the variational classifier.

		Args:
			W (array[float]): 2-d array of variables for one layer

		"""

		# W = W.numpy()

		# Entanglement Layer

		for i in range(self.num_of_wires):
			qml.CNOT(wires=[i, (i + 1) % self.num_of_wires])

		if self.more_entangle == True:
			for j in range(self.num_of_wires):
				qml.CNOT(wires=[j, (j + 2) % self.num_of_wires])

		# Rotation Layer
		for k in range(self.num_of_wires):
			qml.Rot(W[k, 0], W[k, 1], W[k, 2], wires=k)

	def _statepreparation(self, angles):
		if self.hadamard_gate == True:
			for i in range(self.num_of_input):
				qml.Hadamard(wires=i)
				
		for i in range(self.num_of_input):
			qml.RY(angles[i,0], wires=i)
			qml.RZ(angles[i,1], wires=i)

# For noisy simulation
class VQCVariationalLoadingFlexNoisy(VQCBaseClass):
	def __init__(
			self,
			num_of_input= 10,
			num_of_output= 4,
			num_of_wires = 10,
			num_of_layers = 2,
			var_Q_circuit = None,
			var_Q_bias = None,
			qdevice = "default.qubit",
			hadamard_gate = False,
			more_entangle = False,
			gpu = False,
			noisy_dev = None):
		super().__init__(num_of_input, num_of_output, num_of_wires, num_of_layers, var_Q_circuit, var_Q_bias, qdevice, gpu)
		self.hadamard_gate = hadamard_gate
		self.more_entangle = more_entangle
		self.dev = noisy_dev

	def _layer(self, W):
		""" Single layer of the variational classifier.

		Args:
			W (array[float]): 2-d array of variables for one layer

		"""

		# W = W.numpy()

		# Entanglement Layer

		for i in range(self.num_of_wires):
			qml.CNOT(wires=[i, (i + 1) % self.num_of_wires])

		if self.more_entangle == True:
			for j in range(self.num_of_wires):
				qml.CNOT(wires=[j, (j + 2) % self.num_of_wires])

		# Rotation Layer
		for k in range(self.num_of_wires):
			qml.Rot(W[k, 0], W[k, 1], W[k, 2], wires=k)

	def _statepreparation(self, angles):
		if self.hadamard_gate == True:
			for i in range(self.num_of_input):
				qml.Hadamard(wires=i)
				
		for i in range(self.num_of_input):
			qml.RY(angles[i,0], wires=i)
			qml.RZ(angles[i,1], wires=i)


class VQCBasisLoadingFlex(VQCBaseClass):
	def __init__(
			self,
			num_of_input= 10,
			num_of_output= 4,
			num_of_wires = 10,
			num_of_layers = 2,
			var_Q_circuit = None,
			var_Q_bias = None,
			qdevice = "default.qubit",
			hadamard_gate = False,
			more_entangle = False,
			gpu = False):
		super().__init__(num_of_input, num_of_output, num_of_wires, num_of_layers, var_Q_circuit, var_Q_bias, qdevice, gpu)
		self.hadamard_gate = hadamard_gate
		self.more_entangle = more_entangle

	def _layer(self, W):
		""" Single layer of the variational classifier.

		Args:
			W (array[float]): 2-d array of variables for one layer

		"""

		# W = W.numpy()

		# Entanglement Layer

		for i in range(self.num_of_wires):
			qml.CNOT(wires=[i, (i + 1) % self.num_of_wires])

		if self.more_entangle == True:
			for j in range(self.num_of_wires):
				qml.CNOT(wires=[j, (j + 2) % self.num_of_wires])

		# Rotation Layer
		for k in range(self.num_of_wires):
			qml.Rot(W[k, 0], W[k, 1], W[k, 2], wires=k)

	def _statepreparation(self, angles):
		# Here the angles = [1,0,1,1] (binary numbers)
		if self.hadamard_gate == True:
			for i in range(self.num_of_input):
				qml.Hadamard(wires=i)
				
		for i in range(self.num_of_input):
			qml.RX(np.pi * angles[i], wires=i)
			qml.RZ(np.pi * angles[i], wires=i)


class VQCHybridLoadingFlex(VQCBaseClass):
	def __init__(
			self,
			num_of_basis_encoding = 4,
			num_of_variational_encoding = 4,
			num_of_input= 10,
			num_of_output= 4,
			num_of_wires = 10,
			num_of_layers = 2,
			var_Q_circuit = None,
			var_Q_bias = None,
			qdevice = "default.qubit",
			hadamard_gate = False,
			more_entangle = False,
			gpu = False):
		super().__init__(num_of_input, num_of_output, num_of_wires, num_of_layers, var_Q_circuit, var_Q_bias, qdevice, gpu)
		self.hadamard_gate = hadamard_gate
		self.more_entangle = more_entangle

		self.num_of_basis_encoding = num_of_basis_encoding
		self.num_of_variational_encoding = num_of_variational_encoding

	def _layer(self, W):
		""" Single layer of the variational classifier.

		Args:
			W (array[float]): 2-d array of variables for one layer

		"""

		# W = W.numpy()

		# Entanglement Layer

		# The num_of_basis_encoding == num_variational_encoding

		# for i in range(0, self.num_of_basis_encoding):
		# 	qml.CNOT(wires=[i, (i + self.num_of_basis_encoding) % self.num_of_wires])

		# for i in range(self.num_of_basis_encoding, self.num_of_basis_encoding + self.num_of_variational_encoding):
		# 	qml.CNOT(wires=[i, (i + 1) % self.num_of_wires])

		for i in range(self.num_of_wires):
			qml.CNOT(wires=[i, (i + 1) % self.num_of_wires])

		if self.more_entangle == True:
			for j in range(self.num_of_wires):
				qml.CNOT(wires=[j, (j + 2) % self.num_of_wires])

		# Rotation Layer
		for k in range(self.num_of_wires):
			qml.Rot(W[k, 0], W[k, 1], W[k, 2], wires=k)

	def _statepreparation(self, angles):
		# Here the angles = [1,0,1,1] (binary numbers)
		if self.hadamard_gate == True:
			for i in range(self.num_of_basis_encoding, self.num_of_basis_encoding + self.num_of_variational_encoding):
				qml.Hadamard(wires=i)
				
		for i in range(0, self.num_of_basis_encoding):
			qml.RX(np.pi * angles[i,0], wires=i)
			qml.RZ(np.pi * angles[i,0], wires=i)

		for i in range(self.num_of_basis_encoding, self.num_of_basis_encoding + self.num_of_variational_encoding):
			qml.RY(angles[i,0], wires=i)
			qml.RZ(angles[i,1], wires=i)



class VQCVariationalLoadingDoubleEntangle(VQCBaseClass):
	def __init__(
			self,
			num_of_input= 10,
			num_of_output= 4,
			num_of_wires = 10,
			num_of_layers = 2,
			var_Q_circuit = None,
			var_Q_bias = None,
			qdevice = "default.qubit",
			gpu = False):
		super().__init__(num_of_input, num_of_output, num_of_wires, num_of_layers, var_Q_circuit, var_Q_bias, qdevice, gpu)

	def _layer(self, W):
		""" Single layer of the variational classifier.

		Args:
			W (array[float]): 2-d array of variables for one layer

		"""

		# W = W.numpy()

		# Entanglement Layer

		for i in range(self.num_of_wires):
			qml.CNOT(wires=[i, (i + 1) % self.num_of_wires])

		for j in range(self.num_of_wires):
			qml.CNOT(wires=[j, (j + 2) % self.num_of_wires])

		# Rotation Layer
		for k in range(self.num_of_wires):
			qml.Rot(W[k, 0], W[k, 1], W[k, 2], wires=k)

	def _statepreparation(self, angles):
		for i in range(self.num_of_input):
			qml.RY(angles[i,0], wires=i)
			qml.RZ(angles[i,1], wires=i)

class VQCVariationalLoadingHadamard(VQCBaseClass):
	def __init__(
			self,
			num_of_input= 10,
			num_of_output= 4,
			num_of_wires = 10,
			num_of_layers = 2,
			var_Q_circuit = None,
			var_Q_bias = None,
			qdevice = "default.qubit",
			gpu = False):
		super().__init__(num_of_input, num_of_output, num_of_wires, num_of_layers, var_Q_circuit, var_Q_bias, qdevice, gpu)

	def _statepreparation(self, angles):
		for i in range(self.num_of_input):
			qml.Hadamard(wires=i)
		for i in range(self.num_of_input):
			qml.RY(angles[i,0], wires=i)
			qml.RZ(angles[i,1], wires=i)


# Variational Encoding with multiple copy to construct tensor product
class VQCVariationalLoadingTensorProduct(VQCBaseClass):
	def __init__(
			self,
			num_of_input= 8,
			num_of_output= 4,
			num_of_wires = 8,
			num_of_layers = 2,
			var_Q_circuit = None,
			var_Q_bias = None,
			qdevice = "default.qubit",
			gpu = False):
		super().__init__(num_of_input, num_of_output, num_of_wires, num_of_layers, var_Q_circuit, var_Q_bias, qdevice, gpu)

	def _statepreparation(self, angles):
		for i in range(self.num_of_input):
			qml.RY(angles[i,0], wires=i)
			qml.RZ(angles[i,1], wires=i)
		for i in range(self.num_of_input):
			qml.RY(angles[i,0], wires=i + self.num_of_input)
			qml.RZ(angles[i,1], wires=i + self.num_of_input)

class VQCVariationalLoadingTensorProductHadamard(VQCBaseClass):
	def __init__(
			self,
			num_of_input= 8,
			num_of_output= 4,
			num_of_wires = 8,
			num_of_layers = 2,
			var_Q_circuit = None,
			var_Q_bias = None,
			qdevice = "default.qubit",
			gpu = False):
		super().__init__(num_of_input, num_of_output, num_of_wires, num_of_layers, var_Q_circuit, var_Q_bias, qdevice, gpu)

	def _statepreparation(self, angles):
		for i in range(self.num_of_wires):
			qml.Hadamard(wires = i)
		for i in range(self.num_of_input):
			qml.RY(angles[i,0], wires=i)
			qml.RZ(angles[i,1], wires=i)
		for i in range(self.num_of_input):
			qml.RY(angles[i,0], wires=i + self.num_of_input)
			qml.RZ(angles[i,1], wires=i + self.num_of_input)


class VQCVariationalLoadingTensorProduct3(VQCBaseClass):
	def __init__(
			self,
			num_of_input= 12,
			num_of_output= 4,
			num_of_wires = 12,
			num_of_layers = 2,
			var_Q_circuit = None,
			var_Q_bias = None,
			qdevice = "default.qubit",
			gpu = False):
		super().__init__(num_of_input, num_of_output, num_of_wires, num_of_layers, var_Q_circuit, var_Q_bias, qdevice, gpu)

	def _statepreparation(self, angles):
		for i in range(self.num_of_input):
			qml.RY(angles[i,0], wires=i)
			qml.RZ(angles[i,1], wires=i)
		for i in range(self.num_of_input):
			qml.RY(angles[i,0], wires=i + self.num_of_input)
			qml.RZ(angles[i,1], wires=i + self.num_of_input)
		for i in range(self.num_of_input):
			qml.RY(angles[i,0], wires=i + 2 * self.num_of_input)
			qml.RZ(angles[i,1], wires=i + 2 * self.num_of_input)




class VQCAmplLoading(VQCBaseClass):
	def __init__(
			self,
			num_of_input= 10,
			num_of_output= 4,
			num_of_wires = 10,
			num_of_layers = 2,
			var_Q_circuit = None,
			var_Q_bias = None,
			qdevice = "default.qubit",
			gpu = False):
		super().__init__(num_of_input, num_of_output, num_of_wires, num_of_layers, var_Q_circuit, var_Q_bias, qdevice, gpu)

	def _statepreparation(self, angles):
		qml.QubitStateVector(angles, wires=list(range(self.num_of_wires)))

	def _angletransformation(self, angles):
		return angles / torch.clamp(torch.sqrt(torch.sum(angles ** 2)), min = 1e-9)






# Not functioning
class ClassicalScaling(VQCBaseClass):
	def forward(self, angles):
		# print(self.var_Q_circuit)
		# print(angles.dtype)
		return self.var_Q_circuit * angles

# For the testing of the classical preprocessing (scaling)
# the scaling params should be differentiable
# self.var_C_scaling

class VariationalQuantumClassifierAmplitudeLoadingBlock_M_IN_N_OUT_CLASSICAL_SCALING:
	def __init__(
			self,
			num_of_input= 10,
			num_of_output= 4,
			num_of_wires = 10,
			num_of_layers = 2,
			var_Q_circuit = None,
			var_Q_bias = None,
			var_C_scaling = None):

		self.var_Q_circuit = var_Q_circuit
		self.var_Q_bias = var_Q_bias
		self.var_C_scaling = var_C_scaling
		self.num_of_input = num_of_input
		self.num_of_output = num_of_output
		self.num_of_wires = num_of_wires
		self.num_of_layers = num_of_layers


		self.dev = qml.device('default.qubit', wires = num_of_wires)

	def set_params(self, var_Q_circuit, var_Q_bias):
		self.var_Q_circuit = var_Q_circuit
		self.var_Q_bias = var_Q_bias

	def init_params(self):
		self.var_Q_circuit = Variable(torch.tensor(0.01 * np.random.randn(self.num_of_layers, self.num_of_wires, 3), device=device).type(dtype), requires_grad=True)
		return self.var_Q_circuit

	def _layer(self, W):
		""" Single layer of the variational classifier.

		Args:
			W (array[float]): 2-d array of variables for one layer

		"""

		# W = W.numpy()

		# Entanglement Layer

		for i in range(self.num_of_wires):
			qml.CNOT(wires=[i, (i + 1) % self.num_of_wires])

		# Rotation Layer
		for j in range(self.num_of_wires):
			qml.Rot(W[j, 0], W[j, 1], W[j, 2], wires=j)

	def circuit(self, angles):

		@qml.qnode(self.dev, interface='torch')
		def _circuit(var_Q_circuit, angles = None):
			"""The circuit of the variational classifier."""

			qml.QubitStateVector(angles, wires=list(range(self.num_of_wires)))

			weights = var_Q_circuit
			
			for W in weights:
				self._layer(W)

			return [qml.expval(qml.PauliZ(k)) for k in range(self.num_of_output)]

		return _circuit(self.var_Q_circuit, angles = angles)

	def _forward(self, angles):
		"""The variational classifier."""

		# weights = self.var_Q_circuit
		bias = self.var_Q_bias 

		# angles = angles / torch.sqrt(torch.sum(angles ** 2))

		angles = self.var_C_scaling * angles

		angles = angles / torch.clamp(torch.sqrt(torch.sum(angles ** 2)), min = 1e-9)

		# print("ANGLES: ", angles)

		# print("ANGLES in VC:")
		# print(angles)

		# raw_output = torch.tensor(circuit(weights, angles=angles), dtype = torch.double)
		raw_output = self.circuit(angles)

		# m = nn.Softmax(dim=0)

		# normalized_output = torch.max(raw_output + bias, torch.tensor([1e-9,1e-9,1e-9,1e-9], device=device).type(dtype))

		# output = m(normalized_output)
		
		return raw_output

	def forward(self, angles):
		# Output torch tensor
		# angles = angles[0]
		# angles = torch.from_numpy(angles)
		# fw = np.array([self._forward(angles).detach().numpy()])
		# fw = self._forward(angles).detach()
		fw = self._forward(angles)
		# print("FW:",fw)
		return fw




class VariationalQuantumClassifierAmplitudeLoadingBlock_M_IN_N_OUT:
	def __init__(
			self,
			num_of_input= 10,
			num_of_output= 4,
			num_of_wires = 10,
			num_of_layers = 2,
			var_Q_circuit = None,
			var_Q_bias = None,
			qdevice = "default.qubit",
			gpu = False):

		self.var_Q_circuit = var_Q_circuit
		self.var_Q_bias = var_Q_bias
		self.num_of_input = num_of_input
		self.num_of_output = num_of_output
		self.num_of_wires = num_of_wires
		self.num_of_layers = num_of_layers
		self.qdevice = qdevice


		if gpu == True and qdevice == "qulacs.simulator":
			self.dev = qml.device(self.qdevice, wires = num_of_wires, gpu = True)
		else:
			self.dev = qml.device(self.qdevice, wires = num_of_wires)

		# self.dev = qml.device(self.qdevice, wires = num_of_wires)

	def set_params(self, var_Q_circuit, var_Q_bias):
		self.var_Q_circuit = var_Q_circuit
		self.var_Q_bias = var_Q_bias
		

	def init_params(self):
		self.var_Q_circuit = Variable(torch.tensor(0.01 * np.random.randn(self.num_of_layers, self.num_of_wires, 3), device=device).type(dtype), requires_grad=True)
		return self.var_Q_circuit

	def _layer(self, W):
		""" Single layer of the variational classifier.

		Args:
			W (array[float]): 2-d array of variables for one layer

		"""

		# W = W.numpy()

		# Entanglement Layer

		for i in range(self.num_of_wires):
			qml.CNOT(wires=[i, (i + 1) % self.num_of_wires])

		# Rotation Layer
		for j in range(self.num_of_wires):
			qml.Rot(W[j, 0], W[j, 1], W[j, 2], wires=j)

	def circuit(self, angles):

		@qml.qnode(self.dev, interface='torch')
		def _circuit(var_Q_circuit, angles = None):
			"""The circuit of the variational classifier."""

			qml.QubitStateVector(angles, wires=list(range(self.num_of_wires)))

			weights = var_Q_circuit
			
			for W in weights:
				self._layer(W)


			return [qml.expval(qml.PauliZ(k)) for k in range(self.num_of_output)]

		return _circuit(self.var_Q_circuit, angles = angles)

	def _forward(self, angles):
		"""The variational classifier."""

		# weights = self.var_Q_circuit
		bias = self.var_Q_bias 

		# angles = angles / torch.sqrt(torch.sum(angles ** 2))

		angles = angles / torch.clamp(torch.sqrt(torch.sum(angles ** 2)), min = 1e-9)

		# print("ANGLES: ", angles)

		# print("ANGLES in VC:")
		# print(angles)

		# raw_output = torch.tensor(circuit(weights, angles=angles), dtype = torch.double)
		raw_output = self.circuit(angles)

		# m = nn.Softmax(dim=0)

		# normalized_output = torch.max(raw_output + bias, torch.tensor([1e-9,1e-9,1e-9,1e-9], device=device).type(dtype))

		# output = m(normalized_output)
		
		return raw_output

	def forward(self, angles):
		# Output torch tensor
		# angles = angles[0]
		# angles = torch.from_numpy(angles)
		# fw = np.array([self._forward(angles).detach().numpy()])
		# fw = self._forward(angles).detach()
		fw = self._forward(angles)
		# print("FW:",fw)
		return fw

class VariationalQuantumClassifierInterBlock_M_IN_N_OUT:
	def __init__(
			self,
			num_of_input= 10,
			num_of_output= 4,
			num_of_wires = 10,
			num_of_layers = 2,
			var_Q_circuit = None,
			var_Q_bias = None,
			qdevice = "default.qubit",
			hadamard_gate = False,
			more_entangle = False,
			gpu = False):

		self.var_Q_circuit = var_Q_circuit
		self.var_Q_bias = var_Q_bias
		self.num_of_input = num_of_input
		self.num_of_output = num_of_output
		self.num_of_wires = num_of_wires
		self.num_of_layers = num_of_layers

		self.qdevice = qdevice

		self.hadamard_gate = hadamard_gate
		self.more_entangle = more_entangle

		if gpu == True and qdevice == "qulacs.simulator":
			self.dev = qml.device(self.qdevice, wires = num_of_wires, gpu = True)
		else:
			self.dev = qml.device(self.qdevice, wires = num_of_wires)

		# self.dev = qml.device(self.qdevice, wires = num_of_wires)

	def set_params(self, var_Q_circuit, var_Q_bias):
		self.var_Q_circuit = var_Q_circuit
		self.var_Q_bias = var_Q_bias

	def init_params(self):
		self.var_Q_circuit = Variable(torch.tensor(0.01 * np.random.randn(self.num_of_layers, self.num_of_wires, 3), device=device).type(dtype), requires_grad=True)
		return self.var_Q_circuit

	def _statepreparation(self, angles):

		"""Quantum circuit to encode a the input vector into variational params

		Args:
			a: feature vector of rad and rad_square => np.array([rad_X_0, rad_X_1, rad_square_X_0, rad_square_X_1])
		"""

		if self.hadamard_gate == True:
			for i in range(self.num_of_input):
				qml.Hadamard(wires=i)

		for i in range(self.num_of_input):
			qml.RY(angles[i,0], wires=i)
			qml.RZ(angles[i,1], wires=i)

	def _layer(self, W):
		""" Single layer of the variational classifier.

		Args:
			W (array[float]): 2-d array of variables for one layer

		"""

		# W = W.numpy()

		# Entanglement Layer

		for i in range(self.num_of_wires):
			qml.CNOT(wires=[i, (i + 1) % self.num_of_wires])

		if self.more_entangle == True:
			for j in range(self.num_of_wires):
				qml.CNOT(wires=[j, (j + 2) % self.num_of_wires])

		# Rotation Layer
		for j in range(self.num_of_wires):
			qml.Rot(W[j, 0], W[j, 1], W[j, 2], wires=j)

	def circuit(self, angles):

		@qml.qnode(self.dev, interface='torch')
		def _circuit(var_Q_circuit, angles):
			"""The circuit of the variational classifier."""

			self._statepreparation(angles)

			weights = var_Q_circuit
			
			for W in weights:
				self._layer(W)

			return [qml.expval(qml.PauliZ(k)) for k in range(self.num_of_output)]

		return _circuit(self.var_Q_circuit, angles)

	def _forward(self, angles):
		"""The variational classifier."""

		# weights = self.var_Q_circuit
		bias = self.var_Q_bias 

		# angles = angles / torch.sqrt(torch.sum(angles ** 2))

		# angles = angles / torch.clamp(torch.sqrt(torch.sum(angles ** 2)), min = 1e-9)

		# print("ANGLES: ", angles)

		# print("ANGLES in VC:")
		# print(angles)

		# raw_output = torch.tensor(circuit(weights, angles=angles), dtype = torch.double)
		raw_output = self.circuit(angles)

		# m = nn.Softmax(dim=0)

		# normalized_output = torch.max(raw_output + bias, torch.tensor([1e-9,1e-9,1e-9,1e-9], device=device).type(dtype))

		# output = m(normalized_output)
		
		return raw_output

	def forward(self, angles):
		# Output torch tensor

		# angles = angles[0]
		# angles = torch.from_numpy(angles)
		# fw = np.array([self._forward(angles).detach().numpy()])
		# fw = self._forward(angles).detach()
		fw = self._forward(angles)
		# print("FW:",fw)
		return fw

class VariationalQuantumClassifierFinalBlock_M_IN_N_OUT:
	def __init__(
			self,
			num_of_input= 10,
			num_of_output= 4,
			num_of_wires = 10,
			num_of_layers = 2,
			var_Q_circuit = None,
			var_Q_bias = None,
			qdevice = "default.qubit",
			hadamard_gate = False):

		self.var_Q_circuit = var_Q_circuit
		self.var_Q_bias = var_Q_bias
		self.num_of_input = num_of_input
		self.num_of_output = num_of_output
		self.num_of_wires = num_of_wires
		self.num_of_layers = num_of_layers

		self.qdevice = qdevice
		self.hadamard_gate = hadamard_gate
		self.dev = qml.device(self.qdevice, wires = num_of_wires)

	def set_params(self, var_Q_circuit, var_Q_bias):
		self.var_Q_circuit = var_Q_circuit
		self.var_Q_bias = var_Q_bias

	def init_params(self):
		self.var_Q_circuit = Variable(torch.tensor(0.01 * np.random.randn(self.num_of_layers, self.num_of_wires, 3), device=device).type(dtype), requires_grad=True)
		return self.var_Q_circuit
		
	def _statepreparation(self, angles):

		"""Quantum circuit to encode a the input vector into variational params

		Args:
			a: feature vector of rad and rad_square => np.array([rad_X_0, rad_X_1, rad_square_X_0, rad_square_X_1])
		"""

		if self.hadamard_gate == True:
			for i in range(self.num_of_wires):
				qml.Hadamard(wires = i)

		for i in range(self.num_of_input):
			qml.RY(angles[i,0], wires=i)
			qml.RZ(angles[i,1], wires=i)

	def _layer(self, W):
		""" Single layer of the variational classifier.

		Args:
			W (array[float]): 2-d array of variables for one layer

		"""

		# W = W.numpy()

		# Entanglement Layer

		for i in range(self.num_of_wires):
			qml.CNOT(wires=[i, (i + 1) % self.num_of_wires])

		# Rotation Layer
		for j in range(self.num_of_wires):
			qml.Rot(W[j, 0], W[j, 1], W[j, 2], wires=j)

	def circuit(self, angles):

		@qml.qnode(self.dev, interface='torch')
		def _circuit(var_Q_circuit, angles):
			"""The circuit of the variational classifier."""

			self._statepreparation(angles)

			weights = var_Q_circuit
			
			for W in weights:
				self._layer(W)

			return [qml.expval(qml.PauliZ(k)) for k in range(self.num_of_output)]

		return _circuit(self.var_Q_circuit, angles)

	def _forward(self, angles):
		"""The variational classifier."""
		# Output is torch Tensor

		# weights = self.var_Q_circuit
		bias = self.var_Q_bias 

		# angles = angles / torch.sqrt(torch.sum(angles ** 2))


		# angles = angles / torch.clamp(torch.sqrt(torch.sum(angles ** 2)), min = 1e-9)

		# print("ANGLES: ", angles)

		# print("ANGLES in VC:")
		# print(angles)

		# raw_output = torch.tensor(circuit(weights, angles=angles), dtype = torch.double)
		raw_output = self.circuit(angles)

		m = nn.Softmax(dim=0)

		clamp = 1e-9 * torch.ones(self.num_of_output).type(dtype).to(device)

		normalized_output = torch.max(raw_output, clamp)

		output = m(normalized_output)
		
		return output

	def forward(self, angles):
		# Output torch tensor

		# angles = angles[0]
		# angles = torch.from_numpy(angles)
		# fw = np.array([self._forward(angles).detach().numpy()])
		# fw = self._forward(angles).detach()
		fw = self._forward(angles)
		# print("FW:",fw)
		return fw

