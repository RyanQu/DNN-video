
import numpy as np
import sys
import fileinput

"""
dic_id_label['maeb0_si1411_1'] = 'sil'

"""
def load_everything(target_feature_file, num_data = 99999999):
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
        #print 'i:'+str(i)
        #print 'temp[0]:'+str(temp[0])
        #print 'dic_label_num:'+str(dic_label_num[dic_id_label[temp[0]]])
        sys.stdout.flush()
        train_nums[i,dic_label_num[dic_id_label[temp[0]]]] = 1
        train_ids.append(temp[0])

    train_ark_file.close()



    return dic_id_label, dic_label_num, dic_num_label, dic_48_39,train_features, train_nums, train_ids
