import numpy as np
import sys
import fileinput
"""
dic_id_label['maeb0_si1411_1'] = 'sil'
In order to support the usage in theano code, Nick made some modification
and add some other stuff
"""
def load_everything(num_data, validation_data, test_data):
    all_data = 1124823
    if num_data > all_data:
        num_data = all_data
    if validation_data + num_data > all_data:
        validation_data = all_data - num_data
    if test_data > 180406:
        test_data = 180406
    dir_data = '../MLDS_HW1_RELEASE_v1/'

    dic_id_label = {}
    dic_label_num = {}
    dic_num_label = {}
    dic_48_39 = {}

    train_file = open(dir_data+'label/train.lab')
    #for line in fileinput.input(dir_data+'label/train.lab'):
    for line in train_file:
        temp = line.strip().split(',')
        dic_id_label[temp[0]]=temp[1]
    train_file.close()    
    
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
    train_features_ans = np.empty([num_data,]) # Nick, an array to store ans
    train_nums = np.zeros([num_data,48]) # answers?
    train_ids = []
    validation_features = np.empty([validation_data,69]) # Nick
    validation_features_ans = np.empty([validation_data,]) # Nick
    validation_nums = np.zeros([validation_data,48]) # Nick
    validation_ids = []
    test_features = np.empty([test_data,69]) # Nick, the test data
    test_features_ans = np.empty([test_data,]) # Nick
    test_nums = np.zeros([test_data,48]) # Nick
    test_ids = [] # Nick
    
    #fbank: 69 dimension, mfcc:  39 dimension
    train_ark_file = open(dir_data+'fbank/train.ark')
    for i in range(num_data + validation_data):
        line = train_ark_file.readline()
        temp = line.strip().split()
        if i < num_data:
            train_features[i] = np.array(map(float,temp[1:100]))
            sys.stdout.flush()
            train_nums[i,dic_label_num[dic_id_label[temp[0]]]] = 1
            train_features_ans[i] = dic_label_num[dic_id_label[temp[0]]]
            train_ids.append(temp[0])
        elif i >= num_data:
            validation_features[i-num_data] = np.array(map(float,temp[1:100]))
            sys.stdout.flush()
            validation_nums[i-num_data,dic_label_num[dic_id_label[temp[0]]]] = 1
            validation_features_ans[i-num_data] = dic_label_num[dic_id_label[temp[0]]]
            validation_ids.append(temp[0])
    train_ark_file.close() 
    test_ark_file = open(dir_data + 'fbank/test.ark')
    for i in range(test_data):
        line = test_ark_file.readline()
        temp = line.strip().split()
        test_features[i] = np.array(map(float,temp[1:100]))
        sys.stdout.flush()
        # test_nums[i,dic_label_num[dic_id_label[temp[0]]]] = 1
        # test_features_ans[i] = dic_label_num[dic_id_label[temp[0]]]
        test_ids.append(temp[0])
    test_ark_file.close() 

    return train_features, train_features_ans, validation_features, validation_features_ans, test_features, test_features_ans
    # return dic_id_label, dic_label_num, dic_num_label, dic_48_39,train_features, train_features_ans, train_nums, train_ids, validation_features, validation_features_ans, validation_nums, validation_ids, test_features, test_features_ans, test_nums, test_ids
