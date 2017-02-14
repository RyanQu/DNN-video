import numpy as np
import sigmoid
import load_weight
reload(load_weight)
from load_weight import *
from sigmoid import *


def load_everything_1(target_feature_file, num_data = 99999999):
    #file_read = open('/Users/Esther/MLSD/MLDS_HW1_RELEASE_v1/fbank/train.ark', 'r')
    #file_Y = open('Y_data.txt', 'r')

    if num_data > 1124823:
        num_data = 1124823
    all_data = 1124823
    dir_data = '../MLDS_HW1_RELEASE_v1/'


    dic_id_label = {}

    train_file = open(dir_data+'label/train.lab')
    #for line in fileinput.input(dir_data+'label/train.lab'):
    for line in train_file:
        temp = line.strip().split(',')
        dic_id_label[temp[0]]=temp[1]

    train_file.close()


    dic_label_num = {}
    dic_num_label = {}
    dic_48_39 = {}

    dir_file = open(dir_data+'phones/48_39.map')
    i = 0
    for line in dir_file:
        temp = line.strip().split()
        dic_label_num[temp[0]] = i
        dic_num_label[i] = temp[0]
        dic_48_39[temp[0]] = temp[1]
        i+=1
    dir_file.close()

    train_features = np.empty([num_data,69])
    train_nums = np.zeros([num_data,48])
    #train_features = np.empty([5000,69])
    #train_nums = np.zeros([5000,48])
    train_ids = []

    #fbank: 69 dimension
    #mfcc:  39 dimension
    train_ark_file = open(dir_data+target_feature_file)
    for i in range(num_data):
    #for line in fileinput.input(dir_data+'fbank/5000.ark'):
        line = train_ark_file.readline()
        temp = line.strip().split()
        train_features[i] = np.array(map(float,temp[1:100]))
        train_ids.append(temp[0])

    train_ark_file.close()



    return dic_id_label, dic_label_num, dic_num_label, dic_48_39,train_features, train_nums, train_ids





#Speech
target_file = 'fbank/test.ark'
num_data = 180406

[dic_id_label, dic_label_num, dic_num_label, dic_48_39, test_feature, test_nums, test_ids] = load_everything_1(target_file, num_data)

#weight_file = './Output/weight_TEST'
weight_file = './Output/' + 'weight_63000_minute_250'
weight, layer_size = load_weight(weight_file)

num_layer = layer_size.size
output_layer = num_layer - 1

a = range(num_layer)
z = range(num_layer)
#input layer: a[0]



a[0] = np.concatenate((np.ones([num_data ,1]), test_feature), axis=1)
z[1] = a[0].dot( weight[0].T)
a[1] = sigmoid(z[1])

a[1] = np.concatenate((np.ones([num_data, 1]),a[1]), axis=1)
z[2] = a[1].dot(weight[1].T)
a[2] = sigmoid(z[2])
"""
for layer in range(num_layer -1):
    if layer is 0:
        a[0] = np.concatenate((np.ones([num_data ,1]), test_feature), axis=1)
    else:
        if num_data is 1:
            a[layer] = np.concatenate((np.ones([1, 1]),a[layer]), axis=1)
        else:
           	a[layer] = np.concatenate((np.ones([num_data, 1]),a[layer]), axis=1)
    print a[layer].shape
    print weight[layer].shape
    z[layer+1] = a[layer].dot( weight[layer].T)
    a[layer+1] = sigmoid(z[layer])
"""

predict_file = open('./Output/predict.CSV', 'w')
predict_file.write('Id,Prediction\n')
for i in range(num_data):
    predict_file.write('%s,%s\n' %(test_ids[i], dic_48_39[dic_num_label[a[output_layer][i].argmax()]]))

predict_file.close()
