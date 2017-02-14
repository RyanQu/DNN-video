

import numpy as np

def load_weight(weight_file_name):
    weight_file = open(weight_file_name, 'r')
    
    #read the first line: 'layer_size:'
    weight_file.readline()
    
    tmp_layer_size = weight_file.readline().split()
    for i in range(len(tmp_layer_size)):
        tmp_layer_size[i] = float(tmp_layer_size[i])
    layer_size = np.array(tmp_layer_size)
    
    weight_file.readline()
    
    tmp_weight = weight_file.readline().split()
    for i in range(len(tmp_weight)):
        tmp_weight[i] = float(tmp_weight[i])
    
    m = layer_size.shape[0] - 1
    weight = range(m)
    for i in range(m):
        weight[i] = np.zeros([layer_size[i+1], layer_size[i]+1])
    
    count = 0 
    for layer in range(m):
        for i in range(weight[layer].shape[0]):
            for j in range(weight[layer].shape[1]):
                weight[layer][i][j] = tmp_weight[count]
                count += 1
    
    return weight, layer_size