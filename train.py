import sys, argparse
from keras.datasets import fashion_mnist, mnist
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import wandb
import seaborn as sn


################################# Parse Arguments #################################

def parse_arguments():
	parser = argparse.ArgumentParser(description='Training Arguments')
	parser.add_argument('-wp', '--wandb_project', type=str, default='dl_assignment_1', help='Project name used to track experiments in Weights & Biases dashboard')
	parser.add_argument('-we', '--wandb_entity', type=str, default='dl_assignment_1', help='Wandb Entity used to track experiments in the Weights & Biases dashboard.')
	parser.add_argument('-d', '--dataset', type=str, default='fashion_mnist', choices=["mnist", "fashion_mnist"], help='Dataset choice: ["mnist", "fashion_mnist"]')
	parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs to train neural network.')
	parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size used to train neural network.')
	parser.add_argument('-l', '--loss', type=str, default='cross_entropy', choices=["mean_squared_error", "cross_entropy"], help='Loss function choice: ["mean_squared_error", "cross_entropy"]')
	parser.add_argument('-o', '--optimizer', type=str, default='nadam', choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], help='Optimizer choice: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]')
	parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='Learning rate used to optimize model parameters')
	parser.add_argument('-m', '--momentum', type=float, default=0.9, help='Momentum used by momentum and nag optimizers.')
	parser.add_argument('-beta', '--beta', type=float, default=0.9, help='Beta used by rmsprop optimizer')
	parser.add_argument('-beta1', '--beta1', type=float, default=0.9, help='Beta1 used by adam and nadam optimizers.')
	parser.add_argument('-beta2', '--beta2', type=float, default=0.999, help='Beta2 used by adam and nadam optimizers.')
	parser.add_argument('-eps', '--epsilon', type=float, default=1e-10, help='Epsilon used by optimizers.')
	parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0005, help='Weight decay used by optimizers.')
	parser.add_argument('-w_i', '--weight_init', type=str, default='Xavier', choices=["random", "Xavier"], help='Weight initialization choice: ["random", "Xavier"]')
	parser.add_argument('-nhl', '--num_layers', type=int, default=4, help='Number of hidden layers used in feedforward neural network.')
	parser.add_argument('-sz', '--hidden_size', type=int, default=128, help='Number of hidden neurons in a feedforward layer.')
	parser.add_argument('-a', '--activation', type=str, default='ReLU', choices=["identity", "sigmoid", "tanh", "ReLU"], help='Activation function choice: ["identity", "sigmoid", "tanh", "ReLU"]')
	return parser.parse_args()



################################# Plot Categories #################################

def plot_categories():

	(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
	# MetaData of Fashion_Mnist dataset ...
	CLASSES = {
		0:'T-shirt/top',
		1:'Trouser',
		2:'Pullover',
		3:'Dress',
		4:'Coat',
		5:'Sandal',
		6:'Shirt',
		7:'Sneaker',
		8:'Bag',
		9:'Ankle boot'
	}
	ind_of_first_occurance = np.argsort(y_train)
	ind = ind_of_first_occurance[np.searchsorted(y_train, np.arange(0,10,1), sorter=ind_of_first_occurance)]
	fig, ax = plt.subplots(nrows=2, ncols=5)
	for i in range(10):
		ax[i//5, i%5].imshow(x_train[ind[i]], cmap='gray')
		ax[i//5, i%5].set_title(CLASSES[i])

	# wandb.init(project="cs6910_assignment1")
	# wandb.run.name = f'List of Categories'
	# wandb.log({'List of Categories':plt})
	# wandb.finish()
	# fig.suptitle('List of Categories')
	plt.show()






################################# Load Data #################################
def load_data(dataset='fashion_mnist', purpose='train'):
	dataset=dataset.lower()
	purpose=purpose.lower()
	x,x_t,y,y_t = None,None,None,None

	if dataset == 'fashion_mnist':
		(x, y), (x_t, y_t) = fashion_mnist.load_data()
	elif dataset == 'mnist':
		(x, y), (x_t, y_t) = mnist.load_data()

	if purpose == 'train':
		x = x.reshape(x.shape[0], 784) / 255
		y = np.eye(10)[y]
		return x, y
	elif purpose == 'test':
		x_t = x_t.reshape(x_t.shape[0], 784) / 255
		y_t = np.eye(10)[y_t]
		return x_t, y_t





################################# Loss #################################
def calculate_loss(y, y_pred, loss_function):
	ls_fn = loss_function.lower()
	if ls_fn == "mean_squared_error":
		return np.sum((y_pred-y) ** 2) / y.shape[0]
	elif ls_fn == "cross_entropy":
		return (-np.sum(y * np.log(y_pred))) / y.shape[0]





################################# Feed Forward #################################
class FF_NN:

	def __init__(self, param):
		self.hidden_layers = param['hidden_lyrs']
		self.neurons = param['neurons']
		self.input_neurons = param['inpt_sz']
		self.output_neurons = param['oupt_sz']
		self.weights = []
		self.bias = []
		self.activation = param['activation']
		self.output_activation = param['oupt_activation']
		self.weight_initialisation = param['weight_initialisation']

		self.get_weights()
		self.get_bias()


	def get_bias(self):
		for _ in range(self.hidden_layers):
			self.bias.append(np.random.randn(self.neurons))
		self.bias.append(np.random.randn(self.output_neurons))

	def get_weights(self):
		if self.weight_initialisation.lower() == 'random':
			self.weights.append(np.random.randn(self.input_neurons, self.neurons))
			for _ in range(self.hidden_layers-1):
				self.weights.append(np.random.randn(self.neurons, self.neurons))
			self.weights.append(np.random.randn(self.neurons, self.output_neurons))

		else:
			limit = np.sqrt(6/(self.input_neurons + self.neurons))
			self.weights.append(np.random.uniform(low=-limit, high=limit, size=(self.input_neurons, self.neurons)))
			limit = np.sqrt(6/(self.neurons + self.neurons))
			for _ in range(self.hidden_layers-1):
				self.weights.append(np.random.uniform(low=-limit, high=limit, size=(self.neurons, self.neurons)))
			limit = np.sqrt(6/(self.neurons + self.output_neurons))
			self.weights.append(np.random.uniform(low=-limit, high=limit, size=(self.neurons, self.output_neurons)))


	def apply_activation(self, data):
		act = self.activation.lower()
		if act == 'sigmoid':
			data = np.maximum(data, -500)
			data = np.minimum(data, 500)
			return 1/(1+np.exp(-data))
		elif act == 'relu':
			return np.maximum(0,data)
		elif act == 'tanh':
			return np.tanh(data)
		elif act == 'identity':
			return data


	def apply_output_activation(self, data):
		oa = self.output_activation.lower()
		if  oa == 'softmax':
			data = np.maximum(data, -500)
			data = np.minimum(data, 500)
			data = np.exp(data)
			return data/np.sum(data,axis=1).reshape(data.shape[0],1)


	def feed_forward(self, input):
		self.A = [input]
		self.H = [input]

		# hidden layer calculations...
		for i in range(self.hidden_layers):
			self.A.append(self.bias[i] + np.matmul(self.H[-1], self.weights[i]))
			self.H.append(self.apply_activation(self.A[-1]))

		# output layer calculations...
		self.A.append(self.bias[-1] + np.matmul(self.H[-1], self.weights[-1]))
		self.H.append(self.apply_output_activation(self.A[-1]))

		return self.H[-1] # shape of H[-1] = 60000,10   shape of H = layers, 60000, neurons in each layer







################################# Backward Propagation #################################

class BP_NN:

	def __init__(
			self,
			ff_nn:FF_NN,
			param):
		self.ff_nn, self.loss, self.activation, self.output_activation = ff_nn, param['loss_function'], param['activation'], param['oupt_activation']


	def der_actvtn(self, x):
		act = self.activation.lower()
		if act == "sigmoid":
			return x * (1 - x)
		elif act == "tanh":
			return 1 - x ** 2
		elif act == "relu":
			return (x > 0).astype(int)
		elif act == "identity":
			return np.ones(x.shape)

	def der_outpt_actvtn(self, yp):
		act = self.output_activation.lower()
		if act == "softmax":
			return np.diag(yp)-np.outer(yp, yp)

	def der_ls(self, y, yp):
		ls = self.loss.lower()
		if ls == "mean_squared_error":
			return yp-y
		elif ls == "cross_entropy":
			return -y/yp


	def propogate_backward(self, y, y_pred):  # y=60000,10   y_pred=60000,10
		self.d_h, self.d_a, self.delta_weights, self.delta_bias = [], [], [], []
		der_outpt_mat = []

		self.d_h.append(self.der_ls(y, y_pred))
		for i in range(y_pred.shape[0]):
				der_outpt_mat.append(np.matmul(self.der_ls(y[i], y_pred[i]), self.der_outpt_actvtn(y_pred[i])))
		der_outpt_arr = np.array(der_outpt_mat)
		self.d_a.append(der_outpt_arr)
		# self.d_a.append(y_pred-y)

		for i in range(self.ff_nn.hidden_layers, 0, -1):
			self.delta_weights.append(np.matmul(self.ff_nn.H[i].T, self.d_a[-1]))
			self.delta_bias.append(np.sum(self.d_a[-1], axis=0))
			self.d_h.append(np.matmul(self.d_a[-1], self.ff_nn.weights[i].T))
			self.d_a.append(self.d_h[-1] * self.der_actvtn(self.ff_nn.H[i]))

		self.delta_weights.append(np.matmul(self.ff_nn.H[0].T, self.d_a[-1]))
		self.delta_weights.reverse()
		self.delta_bias.append(np.sum(self.d_a[-1], axis=0))
		self.delta_bias.reverse()

		y_sh = y.shape[0]
		for i in range(len(self.delta_bias)):
			self.delta_bias[i] = self.delta_bias[i] / y_sh
			self.delta_weights[i] = self.delta_weights[i] / y_sh
			

		return self.delta_weights, self.delta_bias




################################# Optimizer #################################

class Optimizer():
	def __init__(
			self,
			ff_nn: FF_NN,
			bp_nn: BP_NN,
			param
	):
		self.ff_nn, self.bp_nn, self.lr, self.optimizer, self.momentum, self.decay = ff_nn, bp_nn, param['learning_rate'], param['optimizer'], param['momentum'], param['decay']
		self.B1, self.B2, self.eps, self.t = param['beta1'], param['beta2'], param['epsilon'], 0
		self.b_history = [np.zeros_like(i) for i in self.ff_nn.bias]
		self.b_hm = [np.zeros_like(i) for i in self.ff_nn.bias]
		self.w_history = [np.zeros_like(i) for i in self.ff_nn.weights]
		self.w_hm = [np.zeros_like(i) for i in self.ff_nn.weights]


	def optimize(self, delta_weights, delta_bias):
		opt = self.optimizer.lower()
		if(opt == "sgd"):
			self.SGD(delta_weights, delta_bias)
		elif(opt == "momentum"):
			self.MGD(delta_weights, delta_bias)
		elif(opt == "nesterov"):
			self.NAG(delta_weights, delta_bias)
		elif(opt == "rmsprop"):
			self.RMSPROP(delta_weights, delta_bias)
		elif(opt == "adam"):
			self.ADAM(delta_weights, delta_bias)
		elif(opt == "nadam"):
			self.NADAM(delta_weights, delta_bias)


	def SGD(self, delta_weights, delta_bias):
		for i in range(self.ff_nn.hidden_layers + 1):
			self.ff_nn.weights[i] -= self.lr * (delta_weights[i] + self.ff_nn.weights[i]*self.decay)
			self.ff_nn.bias[i] -= self.lr * (delta_bias[i] + self.ff_nn.bias[i]*self.decay)

	def MGD(self, delta_weights, delta_bias):
		for i in range(self.ff_nn.hidden_layers + 1):
			self.w_history[i] = self.momentum * self.w_history[i] + delta_weights[i]
			self.ff_nn.weights[i] -= self.lr * (self.w_history[i] + self.ff_nn.weights[i]*self.decay)
			self.b_history[i] = self.momentum * self.b_history[i] + delta_bias[i]
			self.ff_nn.bias[i] -= self.lr * (self.b_history[i] + self.ff_nn.bias[i]*self.decay)

	def NAG(self, delta_weights, delta_bias):
		for i in range(self.ff_nn.hidden_layers + 1):
			self.w_history[i] = self.momentum * self.w_history[i] + delta_weights[i]
			self.ff_nn.weights[i] -= self.lr * (self.momentum * self.w_history[i] + delta_weights[i] + self.ff_nn.weights[i]*self.decay)
			self.b_history[i] = self.momentum * self.b_history[i] + delta_bias[i]
			self.ff_nn.bias[i] -= self.lr * (self.momentum * self.b_history[i] + delta_bias[i] + self.ff_nn.bias[i]*self.decay)


	def RMSPROP(self, delta_weights, delta_bias):
		for i in range(self.ff_nn.hidden_layers + 1):
			self.w_history[i] = self.w_history[i]*self.momentum + (1-self.momentum)*delta_weights[i]**2
			self.ff_nn.weights[i] -= delta_weights[i]*(self.lr / (np.sqrt(self.w_history[i]) + self.eps)) + self.decay * self.ff_nn.weights[i] * self.lr
			self.b_history[i] = self.b_history[i]*self.momentum + (1-self.momentum)*delta_bias[i]**2
			self.ff_nn.bias[i] -= delta_bias[i]*(self.lr / (np.sqrt(self.b_history[i]) + self.eps)) + self.decay * self.ff_nn.bias[i] * self.lr


	def ADAM(self, delta_weights, delta_bias):
		for i in range(self.ff_nn.hidden_layers + 1):
			self.w_hm[i] = self.B1 * self.w_hm[i] + (1 - self.B1) * delta_weights[i]
			self.w_history[i] = self.B2 * self.w_history[i] + (1 - self.B2) * delta_weights[i]**2
			self.w_hat_hm = self.w_hm[i] / (1 - self.B1**(self.t + 1))
			self.w_history_hat = self.w_history[i] / (1 - self.B2**(self.t + 1))
			self.ff_nn.weights[i] -= self.lr * (self.w_hat_hm / ((np.sqrt(self.w_history_hat)) + self.eps) + self.decay * self.ff_nn.weights[i])

			self.b_hm[i] = self.B1 * self.b_hm[i] + (1 - self.B1) * delta_bias[i]
			self.b_history[i] = self.B2 * self.b_history[i] + (1 - self.B2) * delta_bias[i]**2
			self.b_hat_hm = self.b_hm[i] / (1 - self.B1**(1+self.t))
			self.h_hat_b = self.b_history[i] / (1 - self.B2**(1+self.t))
			self.ff_nn.bias[i] -= self.lr * (self.b_hat_hm / ((np.sqrt(self.h_hat_b)) + self.eps) + self.decay * self.ff_nn.bias[i])


	def NADAM(self, delta_weights, delta_bias):
		for i in range(self.ff_nn.hidden_layers + 1):
			self.w_hm[i] = self.B1 * self.w_hm[i] + (1 - self.B1) * delta_weights[i]
			self.w_hat_hm = self.w_hm[i] / (1 - self.B1 ** (self.t + 1))
			self.w_history[i] = self.B2 * self.w_history[i] + (1 - self.B2) * delta_weights[i]**2
			self.w_history_hat = self.w_history[i] / (1 - self.B2 ** (self.t + 1))
			w_temp = self.B1 * self.w_hat_hm + ((1 - self.B1) / (1 - self.B1 ** (self.t + 1))) * delta_weights[i]
			self.ff_nn.weights[i] -= self.lr * (w_temp / ((np.sqrt(self.w_history_hat)) + self.eps) + self.decay * self.ff_nn.weights[i])


			self.b_hm[i] = self.B1 * self.b_hm[i] + (1 - self.B1) * delta_bias[i]
			self.b_hat_hm = self.b_hm[i] / (1 - self.B1 ** (self.t + 1))
			self.b_history[i] = self.B2 * self.b_history[i] + (1 - self.B2) * delta_bias[i]**2
			self.h_hat_b = self.b_history[i] / (1 - self.B2 ** (self.t + 1))
			b_temp = self.B1 * self.b_hat_hm + ((1 - self.B1) / (1 - self.B1 ** (self.t + 1))) * delta_bias[i]
			self.ff_nn.bias[i] -= self.lr * (b_temp / ((np.sqrt(self.h_hat_b)) + self.eps) + self.decay * self.ff_nn.bias[i])






################################# Train Function #################################

def train():
	
	wandb.init()
	PARAMETERS = wandb.config
	wandb.run.name = f"lf_{PARAMETERS['loss_function']}_ac_{PARAMETERS['activation']}_opt_{PARAMETERS['optimizer']}_bs_{PARAMETERS['batch_sz']}_wi_{PARAMETERS['weight_initialisation']}_hl_{PARAMETERS['hidden_lyrs']}_sz_{PARAMETERS['neurons']}"
	
	x_train, y_train = load_data(PARAMETERS['dataset'], 'train')
	np.random.seed(7)
	ff_nn = FF_NN(PARAMETERS)
	bp_nn = BP_NN(ff_nn, PARAMETERS)
	opt = Optimizer(ff_nn, bp_nn, PARAMETERS)
	batch_size = PARAMETERS['batch_sz']

	x_train, x_train_t, y_train, y_train_t = train_test_split(x_train, y_train, test_size=0.1, random_state=7)

	for epoch in range(PARAMETERS['epochs']):
		for i in range(0, x_train.shape[0], batch_size):
			y_batch = y_train[i:i+batch_size]
			x_batch = x_train[i:i+batch_size]
			opt.optimize(*bp_nn.propogate_backward(y_batch, ff_nn.feed_forward(x_batch)))

		opt.t += 1
		y_pred = ff_nn.feed_forward(x_train)
		y_pred_t = ff_nn.feed_forward(x_train_t)
		print("epoch-",epoch+1)
		print("accuracy-",np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_train, axis=1)) / y_train.shape[0])
		print("loss-", calculate_loss(y_train, y_pred, PARAMETERS['loss_function']))
		print("val_accuracy-",np.sum(np.argmax(y_pred_t, axis=1) == np.argmax(y_train_t, axis=1)) / y_train_t.shape[0])


		lg={
				'accuracy':np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_train, axis=1)) / y_train.shape[0],
				'val_accuracy':np.sum(np.argmax(y_pred_t, axis=1) == np.argmax(y_train_t, axis=1)) / y_train_t.shape[0],
				'epoch':epoch+1,
				'loss':calculate_loss(y_train, y_pred, PARAMETERS['loss_function']),
				'val_loss':calculate_loss(y_train_t, y_pred_t, PARAMETERS['loss_function'])
		}
		wandb.log(lg)


	return ff_nn


if __name__ == "__main__": 
	inp = vars(parse_arguments())
	sweep_config = {
		"method": "grid",
		"metric": {"goal": "maximize", "name": "val_accuracy"},
		"parameters": {
			"inpt_sz": {"values": [784]},
			"oupt_sz": {"values": [10]},
			"oupt_activation": {"values": ["softmax"]},
			"dataset": {"values": [inp['dataset']]},
			"loss_function": {"values": [inp['loss']]},
			"beta": {"values": [inp['beta']]},
			"beta1": {"values": [inp['beta1']]},
			"beta2": {"values": [inp['beta2']]},
			"neurons": {"values": [inp['hidden_size']]},
			"hidden_lyrs": {"values": [inp['num_layers']]},
			"activation": {"values": [inp['activation']]},
			"learning_rate": {"values": [inp['learning_rate']]},
			"optimizer": {"values": [inp['optimizer']]},
			"momentum": {"values": [inp['momentum']]},
			"batch_sz": {"values": [inp['batch_size']]},
			"epochs": {"values": [inp['epochs']]},
			"weight_initialisation": {"values": [inp['weight_init']]},
			"decay": {"values": [inp['weight_decay']]},
			"epsilon": {"values": [inp['epsilon']]},
		}
	}

	sweep_id = wandb.sweep(sweep_config, project=inp['wandb_project'])
	wandb.agent(sweep_id, function=train)
		


	