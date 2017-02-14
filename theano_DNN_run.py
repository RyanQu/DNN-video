from theano_DNN_class import Network
from theano_DNN_class import load_data_shared
from theano_DNN_class import FullyConnectedLayer
from theano_DNN_class import SoftmaxLayer
from theano_sigmoid import sigmoid

mini_batch_size = 2000
eta = 0.01
layer1_num = 69
layer2_num= 128
layer3_num = 48
epochs = 20
training_data_num = 800000
validation_data_num = 200000
test_data_num = 180000


training_data, validation_data, test_data = load_data_shared(training_data_num, validation_data_num, test_data_num)

"""
layer12 = FullyConnectedLayer(69,128,sigmoid)
layer23 = FullyConnectedLayer(128,48,sigmoid)"""
layer12 = SoftmaxLayer(layer1_num,layer2_num)
layer23 = SoftmaxLayer(layer2_num,layer3_num)

layer_list = []
layer_list.append(layer12)
layer_list.append(layer23)

# make a class
DNN1 = Network(layer_list,mini_batch_size)
DNN1.SGD(training_data, epochs, mini_batch_size, eta, validation_data, test_data)
