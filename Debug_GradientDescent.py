#Debug_GradientDescent
#Author: yesjimmy62
#Usage:
#Add to Main.py:
#   Debug_GradientDescent(X, Y, weight, layer_size)
#(don't forget to import)
#
import random
import CostFunction
reload(CostFunction)
from CostFunction import *
import CostFunction_SpecialY
reload(CostFunction_SpecialY)
from CostFunction_SpecialY import *


calculation_number_limit = 999999

def NumericalGradient(weight, layer_size, X, Y, Lambda):
    numerical_weight_gra = range(len(weight))
    for i in range(len(weight)):
        numerical_weight_gra[i] = np.zeros(weight[i].shape)
        
    epsilon = 0.001
    count = 0
    for layer in range(layer_size.size-1):
        for i in range(weight[layer].shape[0]):
            for j in range(weight[layer].shape[1]):
                #print 'layer:'+ str(layer)
                #print 'weight[layer].shape:'+ str(weight[layer].shape)
                #print 'i:'+ str(i)+',  j:'+str(j)
                tmp_weight = weight[layer][i][j]
                weight[layer][i][j] = tmp_weight + epsilon
                Cost1, weight_gradient = CostFunction(weight, layer_size, X, Y, 1)
                weight[layer][i][j] = tmp_weight - epsilon
                Cost0, weight_gradient = CostFunction(weight, layer_size, X, Y, 1)
                weight[layer][i][j] = tmp_weight
                numerical_weight_gra[layer][i][j] = (Cost1-Cost0)/epsilon/2.
                count += 1
                if count >= calculation_number_limit:
                    return numerical_weight_gra
    return numerical_weight_gra

def Debug_GradientDescent(X, Y, weight, layer_size):
    [row_X, col_X] = X.shape
    Bunch_size = 1
    bunch_sample = random.sample(range(row_X), Bunch_size)
    X_Bunch = X[bunch_sample]
    Y_Bunch = Y[bunch_sample]
    
    Cost, weight_gradient = CostFunction(weight, layer_size, X_Bunch, Y_Bunch, 1)

    numerical_weight_gra = NumericalGradient(weight, layer_size, X_Bunch, Y_Bunch, 1)
    
    print('Debug: GradientDescent')
    count = 0 
    for layer in range(layer_size.size-1):
        for i in range(weight_gradient[layer].shape[0]):
            for j in range(weight_gradient[layer].shape[1]):
                print ('layer:%d , No.:%d->%d \n' %(layer, i, j))
                print ('%10.5e %10.5e\n' %(weight_gradient[layer][i][j], numerical_weight_gra[layer][i][j]))
                
                count += 1
                if count >= calculation_number_limit:
                    return numerical_weight_gra

def NumericalGradient_SpecialY(weight, layer_size, X, Y, Lambda):
    numerical_weight_gra = range(len(weight))
    for i in range(len(weight)):
        numerical_weight_gra[i] = np.zeros(weight[i].shape)
        
    epsilon = 0.001
    count = 0
    for layer in range(layer_size.size-1):
        for i in range(weight[layer].shape[0]):
            for j in range(weight[layer].shape[1]):
                #print 'layer:'+ str(layer)
                #print 'weight[layer].shape:'+ str(weight[layer].shape)
                #print 'i:'+ str(i)+',  j:'+str(j)
                tmp_weight = weight[layer][i][j]
                weight[layer][i][j] = tmp_weight + epsilon
                Cost1, weight_gradient = CostFunction_SpecialY(weight, layer_size, X, Y, 1)
                weight[layer][i][j] = tmp_weight - epsilon
                Cost0, weight_gradient = CostFunction_SpecialY(weight, layer_size, X, Y, 1)
                weight[layer][i][j] = tmp_weight
                numerical_weight_gra[layer][i][j] = (Cost1-Cost0)/epsilon/2.
                
                count+=1
                if count >= calculation_number_limit:
                    return numerical_weight_gra
    return numerical_weight_gra

def Debug_GradientDescent_SpecialY(X, Y, weight, layer_size):
    [row_X, col_X] = X.shape
    Bunch_size = 1
    bunch_sample = random.sample(range(row_X), Bunch_size)
    X_Bunch = X[bunch_sample]
    Y_Bunch = Y[bunch_sample]
    
    Cost, weight_gradient = CostFunction_SpecialY(weight, layer_size, X_Bunch, Y_Bunch, 1)

    numerical_weight_gra = NumericalGradient_SpecialY(weight, layer_size, X_Bunch, Y_Bunch, 1)
    
    print('Debug: GradientDescent')
    count = 0
    for layer in range(layer_size.size-1):
        for i in range(weight_gradient[layer].shape[0]):
            for j in range(weight_gradient[layer].shape[1]):
                print ('layer:%d , No.:%d->%d \n' %(layer, i, j))
                print ('%10.5e %10.5e\n' %(weight_gradient[layer][i][j], numerical_weight_gra[layer][i][j]))
                
                count += 1
                if count >= calculation_number_limit:
                    return 
    