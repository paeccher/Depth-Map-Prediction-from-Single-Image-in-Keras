from keras import *
from sklearn.model_selection import train_test_split

from model import *
from generator import *
from utils import *
from synch import *
from loss import *
from error import *

import tensorflow as tf
import argparse


def parseArgs():
    parser = argparse.ArgumentParser(description='Monocular RGB depth prediction')
    parser.add_argument('--epochs', help='number of epochs to train', default=1, type=int)
    parser.add_argument('--lr', help='learning rate', default='0.0001', type=float)
    parser.add_argument('--beta_1', default='0.9', type=float)
    parser.add_argument('--beta_2', default='0.999', type=float)
    parser.add_argument('--decay', default='0.0', type=float)
    parser.add_argument('--amsgrad', default=False, type=bool)
    parser.add_argument('--batchSize', help='batch size', default='32',type=int)
    parser.add_argument('--nameOfTheModel', help='the name to use when saving the model', default='modelScale12.hdf5', type=str)
    args = parser.parse_args()
    return args

def main():

    height=120
    length=160

    args = parseArgs()

    rgb = []
    accel = []
    depth = []

    get_synched_frames(depth, rgb, accel)

    modelScale12 = scale1_2Model()
    epochs_scale12 = args.epochs
    batch_size = args.batchSize
    adam_scale12 = optimizers.Adam(lr=args.lr, beta_1=args.beta_1, beta_2=args.beta_2, decay=args.decay, amsgrad=args.amsgrad)

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                            patience=3,
                                            verbose=1,
                                            mode='auto',
                                            restore_best_weights='true')

    modelScale12.compile(optimizer=adam_scale12, loss=lossWrapper(height,length,batch_size), metrics=[ errorWrapper(height,length,batch_size) ])

    rgb_train, rgb_val, depth_train, depth_val = train_test_split(rgb, depth, test_size=0.3, random_state=1)

    data_generator = keras.preprocessing.image.ImageDataGenerator()

    train_batch = trainingGenerator(rgb_train, depth_train, batch_size, data_generator)
    val_batch = trainingGenerator(rgb_val, depth_val, batch_size, data_generator)

    modelScale12.fit_generator(generator = train_batch,
                        steps_per_epoch = int(len(rgb_train) / batch_size),
                        epochs = epochs_scale12,
                        verbose = 1,
                        validation_data = val_batch,
                        shuffle=True,
                        validation_steps = int(len(rgb_val) / batch_size),
                        callbacks=[early_stop])

    modelScale12.save("models/scale12/"+args.nameOfTheModel)

main()