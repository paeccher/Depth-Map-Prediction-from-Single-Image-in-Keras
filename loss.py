import tensorflow.keras.backend as K 
from utils import *

def lossWrapper(heigth, length, batch_size):
    batch_size = batch_size
    h = heigth
    l = length

    def Ldepth(y_true, y_pred):
        n = l*h*batch_size

        xfil = der_filterX()
        yfil = der_filterY()
        img = K.log( y_true[:,:,:,0] + 1e-4 ) - K.log( y_pred[:,:,:,0] + 1e-4)  #(?,120,160)
        d = K.expand_dims(img,3) #(?,120,160,1)
        firstTer = K.sum(K.mean(K.pow(K.reshape(d,(-1,120*160)),2),1))
        sndTer = K.pow(K.sum(K.reshape(d,(-1,120*160))),2)/(2*(n**2))
        derx = K.conv2d(d, xfil, padding='same') #(?,120,160,1)
        dery = K.conv2d(d, yfil, padding='same') #(?,120,160,1)
        derxp2 = K.pow(derx,2)
        deryp2 = K.pow(dery,2)
        trzTer = K.sum(K.sum(K.sum(K.sum(derxp2 + deryp2))))
        trzTer = trzTer/n
        result = firstTer-sndTer+trzTer
        return result
    
    return Ldepth
