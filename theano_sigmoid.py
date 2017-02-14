import theano
import theano.tensor as T
from theano import function
from theano.tensor import tanh
from theano.tensor.nnet import sigmoid as sig

### set up sigmoid function
z_sigmoid_matrix = T.dvector('z')
sigmoid_calculation = (1 + T.tanh(z_sigmoid_matrix/2)) / 2
sigmoid_result = function([z_sigmoid_matrix], sigmoid_calculation)
 
def sigmoid(z):
    # return sigmoid_result([z]) #[0][0]
    return sig(z) #[0][0]
 
def sigmoid_prime(z):
    """derivative of sigmoid"""
    return sigmoid(z)*(1-sigmoid(z))

