from numpy.random import uniform
import numpy as np

class NeuralNetwork:
	def __init__(self, n_layers=1, n_neurons=4):
		self.w_in = uniform(low = -1, high = 1, size = (2, n_neurons))
		self.b_in = uniform(low = -1, high = 1, size = n_neurons)
		self.n_layers = n_layers
		self.n_neurons = n_neurons
		
		if n_layers > 1:
			self.w_hidden = uniform(low = -3, high = 3, size = (n_layers, n_neurons, n_neurons))
			self.b_hidden = uniform(low = -1, high = 1, size = (n_layers, n_neurons))
		else:
			self.w_hidden = None
			self.b_hidden = None
		self.w_out = uniform(low = -1, high = 1, size = (n_neurons, 1))
		self.b_out = uniform(low = -1, high = 1, size = (1)) 
		
	@staticmethod
	def activate_layer(y_in, w, b):
		z = np.dot(y_in, w) + b 
		s = 1. / (1. + np.exp(-z))
		return s
	
	def feedforward(self, y_in):
		y = self.activate_layer(y_in, self.w_in, self.b_in)
		if self.n_layers > 1:
			for i in range(self.w_hidden.shape[0]):
				y = self.activate_layer(y, self.w_hidden[i], self.b_hidden[i])
		output = self.activate_layer(y, self.w_out, self.b_out)
		return output
	
	def visualize(self, grid_size=50, colormap='viridis', c_reverse=True):
		import matplotlib.pyplot as plt
		import matplotlib as mpl
		import numpy as np

		mpl.rcParams['figure.dpi'] = 300

		# Create grid
		x = np.linspace(-0.5, 0.5, grid_size)
		y = np.linspace(-0.5, 0.5, grid_size)
		xx, yy = np.meshgrid(x, y)

		# Stack coordinates into (N, 2) array
		coords = np.column_stack([xx.ravel(), yy.ravel()])

		# Feedforward
		y_out = self.feedforward(coords)

		# Reshape to grid
		y_out_2d = y_out.reshape(grid_size, grid_size)

		# Choose colormap
		cmap = plt.cm.get_cmap(colormap)
		if c_reverse:
		    cmap = cmap.reversed()

		# Plot
		fig, ax = plt.subplots(figsize=(6, 6))
		im = ax.imshow(
		    y_out_2d,
		    extent=[-0.5, 0.5, -0.5, 0.5],
		    interpolation='nearest',
		    cmap=cmap,
		    origin='lower'
		)
		ax.set_title(f"Depth: {self.n_neurons} x {self.n_layers}")
		plt.show()





input_y = input("Values of the two input neurons separate by a comma, example, 0.15,0.78: ") 

a, b = input_y.split(",")
y_in = np.array([float(a),float(b)])

nn = NeuralNetwork()
print(type(nn.w_in))
print(f'w_in shape: {nn.w_in.shape}')
nn.feedforward(y_in)

print("Let's compare it with a deeper neural network")
user_input = input("Input the number of hideen layers and the number of neurons, separated by a comma, example, 20,100: ")
numbre_layer, number_neurons = user_input.split(",")

deep_nn = NeuralNetwork(n_layers=int(numbre_layer), n_neurons=int(number_neurons))
print(type(deep_nn.w_in))
print(f'w_in shape: {deep_nn.w_in.shape}')
print(f'w_hidden shape: {deep_nn.w_hidden.shape}')

deep_nn.feedforward(y_in)

nn.visualize(grid_size=512)
deep_nn.visualize(grid_size=512)
