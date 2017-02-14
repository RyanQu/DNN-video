

def Output_Weight(weight, layer_size, filename):
    
    #file_output = open('./Output/'(info['filename'], 'w')
    file_output = open('./Output/'+filename, 'w')
    
    file_output.write('layer size:\n')
    for num in layer_size:
        file_output.write('%5d' %(num))
    
    file_output.write('\nweight:\n')
    for layer in range(len(weight)):
        for i in range(weight[layer].shape[0]):
            for j in range(weight[layer].shape[1]):
                file_output.write('%6g    ' %(weight[layer][i][j]))
    file_output.close()
