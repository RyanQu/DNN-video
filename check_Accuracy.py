import load_everything
import Accuracy_SpecialY
import load_weight
from load_weight import *
from load_everything import *
from Accuracy_SpecialY import *

target_file = 'fbank/train.ark'
load_data = 1000000
[dic_id_label, dic_label_num, dic_num_label, dic_48_39,train_features, train_nums, train_ids] = load_everything(target_file, load_data)


weight_file = './Output/weight_4000_minute_8.59644608498'
weight, layer_size = load_weight(weight_file)


num_layer = layer_size.size
output_layer = num_layer - 1


[TotalCost, TotalAccuracy] = Accuracy_SpecialY(weight, layer_size, train_features, train_nums, 1)
print '  TotalCost:' +str(TotalCost)
print '  Accuracy:'  +str(TotalAccuracy)
