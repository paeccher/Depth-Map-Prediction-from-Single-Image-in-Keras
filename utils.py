import tensorflow.keras.backend as K 

# filter should have size of:
# [filter_height, filter_width, in_channels, out_channels]
def der_filterX(): 
    return  K.variable([   [[[-1]],[[0]],[[1]]],   [[[-2]],[[0]],[[2]]],  [[[-1]],[[0]],[[1]]]     ]) # 3,3,1,1

def der_filterY():
    return  K.variable( [   [[[-1]],[[-2]],[[-1]]],   [[[0]],[[0]],[[0]]],  [[[1]],[[2]],[[1]]]  ]) # 3,3, 1,1

