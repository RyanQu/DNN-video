""" DNNTheano.py
The DNN network implemting theano
"""
# standard library
import cPickle # turn python object into serial bytes in C

# Third-party libraries
import numpy as np
import theano
import theano.tensor as T
from theano.tensor import shared_randomstreams
from theano.tensor.signal import downsample #?????

# Activation functions for neurons
from theano import function
from theano.tensor import tanh

### constants
dir_data = '../MLDS_HW1_RELEASE_v1'
GPU = True # False to use CPU
if GPU
    print "Run under GPU!"
    try: theano.config.device = 'gpu'
    except: pass # already set
    theano.config.floatX = 'float32'
else:
    print "Run under CPU"

### Load Data ###############################
def load_data_shared():
    train_ark_file = open(dir_data + "fbank.train.ark")



### Main class to construct DNN
class DNNTheano():
    def __init__(self ,layers, mini_batch_size):
        """ takes a 'list' of layers, ex [38,129,49] fully connected
        layer, and value of batch size """
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        # params between layers
        self.x = T.matrix("x") 
        self.y = T.ivector("y")
        init_layer = self.layers[0] # the first layer size
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size) # set layer input
        for j in xrange(1, le(self.layers)):
            prev_layer, layer = self.layers[j-1], self.layers[j]
            # set the input and other stuff of this layer
            # from the previous layer
            layer.set_inpt(prev.layer.output,m prev_layer.output_dropout, self.mini_batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout

    def SGD(self, training_data, epochs, mini_batch_size, eta, 
            validation_data, test_data, lmbda = 0.0):
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data
        # compute number of batched for training, validation and testing
        num_training_batches = size(training_data)/mini_batch_size
        num_validation_batches = size(validation_data)/mini_batch_size
        num_test_batches = size(test_data)/mini_batch_size
        # define the (regularized) cost function, symbolic gradients,
        # and updates
        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])
        cost = self.layers[-1].cost(self)+\
                0.5 * lmbda * l2_norm_squared/num_training_batches
        grads = T.grad(cost, self.params)
        updates = [(param, param-eta*grad) 
                for param, grad in zip(self.params, grads)]




## Define different layer type classes
class FullyConnectedLayer():
    def __init__(self, n_in, n_out, activation_fn = sigmoid, p_dropout = 0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout # omit for now
        # initialize shared weights
        self.w = theano.shared(
            np.asarray(
                np.random.normal(
                    loc = 0.0, scale = np.sqrt(1.0/n_out, size = (n_in, n_out)),
                dtype = theano.config.floatX),
            name = 'w', borrow = True)
        # initialize shared biases
        self.b = theano.shared(
            np.random.normal(loc = 0/0, scale = 1.0, size = (n_out,)),
                dtype = theano.config.floatX),
            name = 'b', borrow = True)
        # store the weights and biases into params
        self.params = [self.w, self.b]

    def set_inpt(self, impt,impt_dropout,mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output =self.activation_fn(
            (1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1) # axis ???
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b)

    def accuracy(self, y):
        """Return the accuracy for the mini-batch"""
        return T.mean(T.eq(y, self.y_out))

### Miscellanea
def size(data):
    "Return the size of the dataset 'data'"
    return data[0].get_value(borrow=True).shape[0]

z_sigmoid_matrix = T.dmatrix('z')
sigmoid_calculation = (1 + T.tanh(z_sigmoid_matrix/2)) / 2
sigmoid_result = function([z_sigmoid_matrix], sigmoid_calculation)

def sigmoid(z):
    return sigmoid_result([[z]])[0][0]

def sigmoid_prime(z):
    """derivative of sigmoid"""
    return sigmoid(z)*(1-sigmoid(z))

""" still not sure about the vector version
sigmoid_vec = np.vectorize(sigmoid)
sigmoid_prime_vec = np.vectorize(sigmoid_prime)
"""
