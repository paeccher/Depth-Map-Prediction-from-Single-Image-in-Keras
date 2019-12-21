import tensorflow.keras.backend as K 

def errorWrapper(height, length, batch_size):
    batch_size = batch_size
    h = height
    l = length

    def scaleInvariantError(y_true, y_pred):
        # y  : prediction
        # y* : ground truth
        # 1/2n^2 * sum[i,j] ((log(yi)-log(yj) - (log(y*i)-log(y*j))^2 ) 

        n = l*h*batch_size

        img = K.log(y_true[:,:,:,0] + 1e-4 ) - K.log(y_pred[:,:,:,0]+ 1e-4 ) #(?,120,160)
        d = K.expand_dims(img,3)  #(?,120,160,1)
        sumVal = K.sum(K.sum(K.sum(K.sum(K.pow(d,2)))))/(n)
        sndTer = (K.sum(K.sum(K.sum(K.sum(d))))**2)/((n)**2)    
        result = sumVal-sndTer
        return result
    
    return scaleInvariantError