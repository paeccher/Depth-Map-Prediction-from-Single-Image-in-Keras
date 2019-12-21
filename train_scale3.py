from keras import *
from keras.models import load_model
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
    parser.add_argument('--batchSize', help='batch size', default='16',type=int)
    parser.add_argument('--scale12model', help='name of the Scale 12 model', default='modelScale12.hdf5', type=str)
    parser.add_argument('--nameOfTheModel', help='the name to use when saving the model', default='modelScale3.hdf5', type=str)
    args = parser.parse_args()
    return args

def main():

    heigth=120
    length=160

    args = parseArgs()
    rgb = []
    accel = []
    depth = []
    batch_size = args.batchSize

    get_synched_frames(depth, rgb, accel)
    
    modelScale3 = Scale3_Model()

    batch_size_scale3 = args.batchSize
    epochs_scale3 = args.epochs
    adam_scale3 = optimizers.Adam(lr=args.lr, beta_1=args.beta_1, beta_2=args.beta_2, decay=args.decay, amsgrad=args.amsgrad)

    modelScale12 = load_model("models/scale12/"+args.scale12model, custom_objects={'lossWrapper': lossWrapper, 'Ldepth': lossWrapper(heigth,length,batch_size), 'errorWrapper': errorWrapper, 'scaleInvariantError': errorWrapper(heigth,length,batch_size)})
    modelScale12._make_predict_function()
    
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=3,
                                                verbose=1,
                                                mode='auto',
                                                restore_best_weights='true')

    modelScale3.compile(optimizer=adam_scale3, loss=lossWrapper(heigth,length,batch_size), metrics=[ errorWrapper(heigth,length,batch_size) ])

    rgb_train, rgb_val, depth_train, depth_val = train_test_split(rgb, depth, test_size=0.3, random_state=1)

    train_batch = scale3TrainingGenerator(rgb_train, depth_train, batch_size_scale3, modelScale12)
    val_batch = scale3TrainingGenerator(rgb_val, depth_val, batch_size_scale3, modelScale12)

    modelScale3.fit_generator(generator = train_batch,
                        steps_per_epoch = int(len(rgb_train) / batch_size_scale3),
                        epochs = epochs_scale3,
                        verbose = 1,
                        validation_data = val_batch,
                        shuffle=True,
                        callbacks=[early_stop],
                        validation_steps = int(len(rgb_val) / batch_size_scale3))

    modelScale3.save("models/scale3/"+args.nameOfTheModel)

main()