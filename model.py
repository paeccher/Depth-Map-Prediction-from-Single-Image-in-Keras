import keras
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Concatenate, Dense, Conv2D, Flatten,MaxPooling2D, Dropout, Input, Reshape, UpSampling3D, UpSampling2D

def scale1_2Model():
    input_1 = Input(shape=(240,320,3))
    lay1_1 = Conv2D(96, kernel_size=(11,11),strides=(4,4),padding='valid', activation='relu',input_shape=(240,320,1))(input_1)
    pool1 = MaxPooling2D(pool_size=(2,2))(lay1_1)
    lay1_2 = Conv2D(256, kernel_size=(5,5),strides=(1,1),padding='valid', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size = (2,2))(lay1_2)
    lay1_3 = Conv2D(384, kernel_size=(3,3),strides=(1,1),padding='same', activation='relu')(pool2)
    lay1_4 = Conv2D(384, kernel_size=(3,3),strides=(1,1),padding='same', activation='relu')(lay1_3)
    lay1_5 = Conv2D(256, kernel_size=(3,3),strides=(2,2),padding='valid', activation='relu')(lay1_4)
    
    lay1_9 = Flatten() (lay1_5)

    lay1_6 = Dense(4096, activation='relu' )(lay1_9)
    drop_1 = Dropout(0.5)(lay1_6)
    lay1_7 = Dense(19200, activation='relu' )(drop_1)
    lay1_8 = Reshape(target_shape=(15,20,64))(lay1_7)
    
    upsample1 = UpSampling3D(size=(4,4,1))(lay1_8)

    lay2_1 = Conv2D(96, kernel_size=(9,9), strides=(2,2),padding='same', activation='relu')(input_1)
    pool2_1 = MaxPooling2D(pool_size=(2,2))(lay2_1)

    concat_1 = Concatenate()([upsample1, pool2_1])

    lay2_2 = Conv2D(64, kernel_size=(5,5), strides=(1,1), padding='same', activation='relu' )(concat_1)
    lay2_3 = Conv2D(64, kernel_size=(5,5), strides=(1,1), padding='same', activation='relu')(lay2_2)
    lay2_4 = Conv2D(64, kernel_size=(5,5), strides=(1,1), padding='same', activation='relu')(lay2_3)
    lay2_5 = Conv2D(1, kernel_size=(5,5), strides=(1,1), padding='same', activation='relu')(lay2_4)
    upsample2 = UpSampling3D((2,2,1))(lay2_5)

    model = Model(inputs=[input_1], outputs=[upsample2])
    
    return model


def Scale3_Model():
    input_1_s3 = Input(shape=(240,320,3))
    input_2 = Input(shape=(120,160,1))
    
    lay3_1 = Conv2D(96, kernel_size=(9,9), strides=(1,1), padding='same', activation='relu')(input_1_s3)
    pool3_1 = MaxPooling2D(pool_size=(2,2))(lay3_1)

    concat_2 = Concatenate()([pool3_1, input_2])
    lay3_2 = Conv2D(64, kernel_size=(5,5), strides=(1,1), padding='same', activation='relu')(concat_2)
    lay3_3 = Conv2D(64, kernel_size=(5,5), strides=(1,1), padding='same', activation='relu')(lay3_2)
    lay3_4 = Conv2D(1, kernel_size=(5,5), strides=(1,1), padding='same', activation='relu')(lay3_3)

    model = Model(inputs=[input_1_s3, input_2], outputs=[lay3_4])
    return model