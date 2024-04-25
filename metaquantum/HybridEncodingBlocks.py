from timeit import default_timer as timer

# import numpy as np
from pennylane import numpy as np
import warnings
import torch
import torch.nn as nn 
from torch.autograd import Variable

import pennylane as qml

from CircuitComponents import VQCBaseClass

dtype = torch.cuda.DoubleTensor if torch.cuda.is_available() else torch.DoubleTensor
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class HybridCompBasisVarEncoding(VQCBaseClass):
	def __init__(self):
		'''
		The input format is 
		[h_0, h_1, h_2, h_3, x_0, x_1, x_2, x_3]
		where h_i are in {0, 1}, x_j are floating numbers.
		The inputs are processed outside the circuit.
		the h_i are transformed into angles (multiplied by pi)
		the x_j are transformed into angles (arctan())

		'''



		pass

	def _layer(self):
		'''
		This is for the hybrid computational basis and variational encoding.

		'''
		assert(self.num_of_computational_basis_encoding == self.num_of_variational_encoding)
		
		# Entanglement part

		for i in range(self.num_of_computational_basis_encoding):
			qml.CNOT(wires = [i, i + self.num_of_computational_basis_encoding])
		
		for i in range(self.num_of_variational_encoding):
			qml.CNOT(wires = [self.num_of_computational_basis_encoding + i, self.num_of_computational_basis_encoding + ( (i + 1) % self.num_of_variational_encoding)])

		# Variational / Learning part



		pass

	def _statepreparation(self):
		'''
		First part: Computational Basis encoding
		Second part: Variational Encoding
		'''

		# Computational Basis Encoding
		for i in range(self.num_of_computational_basis_encoding):
			qml.RX(angles_for_x_rot, wires=i)
			qml.RZ(angles_for_z_rot, wires=i)

		# Variational Encoding
		for i in range(self.num_of_computational_basis_encoding, self.num_of_computational_basis_encoding + self.num_of_variational_encoding):
			qml.Hadamard(wires=i)
				
		for i in range(self.num_of_computational_basis_encoding, self.num_of_computational_basis_encoding + self.num_of_variational_encoding):
			qml.RY(angles[i,0], wires=i)
			qml.RZ(angles[i,1], wires=i)

		pass










def main():

	return




if __name__ == '__main__':
	main()




