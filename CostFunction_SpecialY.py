#Cost Function
#log method
"""
input:
    weight:
        #a 3-dim array
        weight[level][row][column]
    layer_size:
        layer_size[0]: input_layer_size
        layer_size[-1]:output_layer_size
    y:
        1-dimensional vector
output:
    Cost:
        the cost calculated by specific cost function
    gradient:
        gradient with respect to weight and bias
"""
import numpy as np
import sigmoid
reload(sigmoid)
from sigmoid import *

def CostFunction_SpecialY(weight, layer_size, X, Y, Lambda):
    #to do: write checking for weight and layer_size
    """
    if not isinstance(weight, np.ndarray):
        print 'ERROR: In CostFunction.py\nWrong type of weight!\n'
        return
    """
    if not isinstance(layer_size, np.ndarray):
        print 'ERROR: In CostFunction.py\nWrong type of weight!\n'
        return

    Cost = 0.
    #to do... what is the weight's type???
    #weight_gradient = np.zeros(weight.shape)
    weight_gradient = range(len(weight))
    for i in range(len(weight)):
        weight_gradient[i] = np.zeros(weight[i].shape)


    num_layer = layer_size.size
    num_weight = num_layer-1
    num_bunch = X.shape[0]   #number of data in a bunch
    #Calculate Cost: forward propagation
    #to do: bunch??, change to adaptive layer number
    a = range(num_layer)
    z = range(num_layer)
    #input layer: a[0]

    for layer in range(num_layer-1):
        if layer is 0:
            a[0] = np.concatenate((np.ones([num_bunch ,1]), X), axis=1)
        else:
            a[layer] = np.concatenate((np.ones([num_bunch, 1]), a[layer]), axis=1)
        z[layer+1] = a[layer].dot(weight[layer].T)
        a[layer+1] = sigmoid(z[layer+1])

    """
    a[0] = X
    for layer in range(num_weight):
        z[layer+1] = a[layer].dot(weight[layer].T)
    """

    for data_index in range(num_bunch):
        #to do...
        Cost = Cost - np.log(a[num_weight][data_index]).dot(Y[data_index]) \
        -np.log(1.-a[num_weight][data_index]).dot(1.-Y[data_index])

    Cost = Cost/num_bunch

    #to do:regularization

    #back-propagation
    #to do: all in matrix form...?
    delta=range(num_layer)
    delta_vec = range(num_layer)
    delta[num_weight] = a[num_weight] - Y

    countdown_list = range(1, num_weight)
    countdown_list.reverse()

    for train_i in range(num_bunch):
        delta_vec[num_weight] = delta[num_weight][train_i].reshape(delta[num_weight][train_i].size,1)

        for layer in countdown_list:
            delta_vec[layer] = np.delete( \
            ((weight[layer].T).dot(delta_vec[layer+1]))*((D_sigmoid(np.concatenate(([1],z[layer][train_i]), axis=1))).reshape(1+layer_size[layer], 1)) \
            , 0, 0)

        for layer in range(num_weight):
            weight_gradient[layer] += delta_vec[layer+1].dot(a[layer][train_i].reshape(1,layer_size[layer]+1))
    #to do:regularization

    #return Cost
    return Cost, weight_gradient


def CostFunction_SpecialY_OPT(bias, weight, layer_size, X, Y, Lambda):
    #to do: write checking for weight and layer_size
    """
    if not isinstance(weight, np.ndarray):
        print 'ERROR: In CostFunction.py\nWrong type of weight!\n'
        return
    """
    if not isinstance(layer_size, np.ndarray):
        print 'ERROR: In CostFunction.py\nWrong type of weight!\n'
        return

    Cost = 0.
    #to do... what is the weight's type???
    #weight_gradient = np.zeros(weight.shape)

    num_layer = layer_size.size
    num_weight = num_layer-1
    num_bunch = X.shape[0]   #number of data in a bunch
    #Calculate Cost: forward propagation


    a = range(num_layer)
    z = range(num_layer)
    a[0] = X
    for layer in range(num_weight):
        z[layer+1] = np.dot(a[layer], weight[layer].T) + np.array([(bias[layer].T)[0] for k in range(num_bunch)])

        a[layer+1] = sigmoid(z[layer+1])

    for data_index in range(num_bunch):
        #to do...
        Cost = Cost - np.log(a[num_weight][data_index]).dot(Y[data_index]) \
        -np.log(1.-a[num_weight][data_index]).dot(1.-Y[data_index])

    Cost = Cost/num_bunch

    #to do:regularization

    #back-propagation
    #to do: all in matrix form...?
    delta=range(num_layer)
    delta[num_weight] = a[num_weight] - Y
    delta[num_weight] = delta[num_weight].T
    countdown_list = range(1, num_weight)
    countdown_list.reverse()

    bias_gradient = range(num_weight)
    weight_gradient = range(num_weight)

    for layer in countdown_list:
        delta[layer] = (weight[layer].T).dot(delta[layer+1])*D_sigmoid(z[layer].T)

    for layer in range(num_weight):
        bias_gradient[layer] = np.sum(delta[layer+1], axis=1).reshape([layer_size[layer+1],1])
        weight_gradient[layer] = delta[layer+1].dot(a[layer])



    #to do:regularization

    return Cost, bias_gradient, weight_gradient

