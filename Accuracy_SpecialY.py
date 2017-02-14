#Accuracy
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

def Accuracy_SpecialY(weight, layer_size, X, Y, Lambda):
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
    
    num_layer = layer_size.size
    num_bunch = X.shape[0]   #number of data in a bunch
    #Calculate Cost: forward propagation
    #to do: bunch??, change to adaptive layer number
    a = range(num_layer)
    z = range(num_layer)
    #input layer: a[0]
    a[0] = np.concatenate((np.ones([num_bunch ,1]), X), axis=1)
    z[1] = a[0].dot( weight[0].T)
    a[1] = sigmoid(z[1])
    
    a[1] = np.concatenate((np.ones([num_bunch, 1]),a[1]), axis=1)
    z[2] = a[1].dot(weight[1].T)
    a[2] = sigmoid(z[2])
    
    num_correct = 0
    for data_index in range(num_bunch):
        Cost = Cost - np.log(a[2][data_index]).dot(Y[data_index])-np.log(1-a[2][data_index]).dot(1-Y[data_index])
        
        #print 'a[2][data_index]:'+str(a[2][data_index])
        #print 'HAHA:'+str(int(a[2][data_index].argmax()))+','+str(label)
        #print 'type(a):'+str(type(int(a[2][data_index].argmax())))+',' +str(type(label))
        
        if int(a[2][data_index].argmax()) is int(Y[data_index].argmax()):
            num_correct += 1
    
    print 'num_correct:'+str(num_correct)+', num_bunch:'+str(num_bunch)
    Accuracy = float(num_correct)/float(num_bunch)
    Cost = Cost/num_bunch
    return [Cost, Accuracy]
    