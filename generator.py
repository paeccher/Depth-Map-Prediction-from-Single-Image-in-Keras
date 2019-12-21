
import keras
import cv2 as cv
import numpy as np
import struct

def isMissingMet(val):
    if ((val <=0) or (val >=8.191)):
        return 1
    else:
        return 0

def convertToMeters(value):
    return 1.0 / (value * -0.0030711016 + 3.3309495161)

def inpaint(img):
    img = np.array(img)
    img.byteswap(inplace=True)
    img = cv.resize(img, (160,120))
    img = convertToMeters(img)
    isImgNoise = [[isMissingMet(e) for e in row] for row in img ]
    isImgNoise = np.array(isImgNoise)
    isImgNoise = isImgNoise.astype('uint8')      
    img = img.astype('float32')
    img = cv.inpaint(img, isImgNoise, 5, cv.INPAINT_NS)
    img = (img/8.191)*255
    img = img.astype('uint8')
    img = img.astype('float')
    return img

class trainingGenerator(keras.utils.Sequence) :

  def __init__(self, image_filenames, labels, batch_size, data_generator) :
    self.image_filenames = image_filenames ##array rgb [String]
    self.depth = labels  #array depth
    self.batch_size = batch_size #Int
    self.data_generator = data_generator
    
  def __len__(self) :
    return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

  def __getitem__(self, idx) :
    #idx : index of current batch
    batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y = self.depth[idx * self.batch_size : (idx+1) * self.batch_size]
    
    rgb_arr = [[cv.imread(('data/' + rgb_name), cv.IMREAD_COLOR)] for rgb_name in batch_x]
    depth_arr = [[cv.imread(('data/' + depth_name), cv.IMREAD_UNCHANGED)] for depth_name in batch_y]
    
    for img in rgb_arr :
        img[0] = cv.resize(img[0], (320,240))

    for img in depth_arr: #batch_y :
        img[0] = inpaint(img[0])

    batch_rgb = np.array( rgb_arr )
    batch_depth = np.array( depth_arr )
    bx = batch_rgb.reshape((-1, 240, 320, 3))
    by = batch_depth.reshape((-1, 120, 160, 1))    
    return (bx, by)


class scale3TrainingGenerator(keras.utils.Sequence) :

  def __init__(self, image_filenames, labels, batch_size, scale_2_model):
    self.image_filenames = image_filenames ##array rgb [String]
    self.depth = labels  #array depth
    self.batch_size = batch_size #Int
    self.scale_2_model = None #scale_2_model
    
  def __len__(self) :
    return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

  def __getitem__(self, idx) :
    #idx : indice del batch corrente
    batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y = self.depth[idx * self.batch_size : (idx+1) * self.batch_size]
    
    rgb_arr = [[cv.imread(('data/' + rgb_name), cv.IMREAD_COLOR)] for rgb_name in batch_x]
    depth_arr = [[cv.imread(('data/' + depth_name), cv.IMREAD_UNCHANGED)] for depth_name in batch_y]

    for img in rgb_arr :
      img[0] = cv.resize(img[0], (320,240))

    for img in depth_arr:#batch_y :
      img[0] = inpaint(img[0])

    batch_rgb = np.array( rgb_arr )
    batch_depth = np.array( depth_arr )

    bx = batch_rgb.reshape((-1, 240, 320, 3))
    by = batch_depth.reshape((-1, 120, 160, 1))

    #with session.as_default():
    #  with session.graph.as_default():
    #    predictions = self.scale_2_model.predict(bx)
    predictions=np.zeros((2,120,160,1))
    bx = [bx,predictions]
    return (bx, by)
