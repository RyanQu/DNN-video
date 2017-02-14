#Neural Network
import numpy as np
import math
import time
import copy
import sys
import load_everything
import ParameterInitialization
import GradientDescent_Bunch_SpecialY
import load_weight
reload(load_everything)
reload(ParameterInitialization)
reload(GradientDescent_Bunch_SpecialY)
reload(load_weight)
from ParameterInitialization import *
from GradientDescent_Bunch_SpecialY import *
from load_everything import *
from load_weight import *


#baseline
#input_layer_size = 69
#hidden_layer_size = 125
#output_layer_size = 48
#layer_size = [input_layer_size, hidden_layer_size, output_layer_size]

#layer_size = [69, 256, 48]
layer_size = [69, 81, 81, 81,81,48]
num_weight_layer = len(layer_size)-1
layer_size = np.array(layer_size)

#loading training data

print 'loading training data: Start...'

time0 = time.time()

target_file = 'fbank/train.ark'
load_data = 10000
[dic_id_label, dic_label_num, dic_num_label, dic_48_39,train_features, train_nums, train_ids] = load_everything(target_file, load_data)

time1 = time.time()
print 'loading training data: End...spend '+str(time1-time0)+ 'sec.'

sys.stdout.flush()

#weight initialization
weight = range(num_weight_layer)
for l in range(num_weight_layer):
    weight[l] = ParameterInitialization(layer_size[l]+1, layer_size[l+1])
weight_copy1 = copy.deepcopy(weight)
weight_copy2 = copy.deepcopy(weight)
learning_rate = 0.0001
iteration =  100000
Bunch_size = 28



#final_weight = GradientDescent(X, Y, weight, layer_size, learning_rate,iteration)

#time0=time.time()
#final_weight = GradientDescent_Bunch_SpecialY(train_features, train_nums, weight_copy1, layer_size, learning_rate, iteration, Bunch_size)
#time1=time.time()
final_weight = GradientDescent_Bunch_SpecialY_OPT(train_features, train_nums, weight_copy2, layer_size, learning_rate, iteration, Bunch_size)
#time2=time.time()

#print 'non_OPT:%f\n' %(time1-time0)
#print 'OPT:%f\n' %(time2-time1)
#final_weight = GradientDescent_Adagrad_SpecialY(train_features, train_nums, weight, layer_size, learning_rate, iteration, Bunch_size)

#Debug_GradientDescent_SpecialY(train_features, train_nums, weight, layer_size)

