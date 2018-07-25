# %matplotlib inline
import os,random
os.environ["KERAS_BACKEND"] = "theano"
#os.environ["THEANO_FLAGS"]  = "device=gpu%d,lib.cnmem=0"%(random.randint(0,3))
import numpy as np
import theano as th
import theano.tensor as T
from keras.utils import np_utils
import keras.models as models
from keras.layers import Input,merge
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten,MaxoutDense
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import *
from keras.layers.wrappers import TimeDistributed
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.layers.recurrent import LSTM
from keras.regularizers import *
from keras.layers.normalization import *
from keras.optimizers import *
from keras.datasets import mnist
import matplotlib.pyplot as plt
import seaborn as sns
import cPickle, random, sys, keras
from keras.models import Model
from IPython import display

sys.path.append("../common")
from keras.utils import np_utils
from tqdm import tqdm

np.set_printoptions(threshold='nan')
img_rows, img_cols = 100, 100

# the data, shuffled and split between train and test sets

# (X_train, y_train), (X_test, y_test) = mnist.load_data()

# X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
# X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train = X_train[0:320]/255
# X_test  = X_train[0:320]/255
import scipy.io as sio
#image = sio.loadmat('linescircle.mat')
#data1=np.abs(image['lines_final'])
WB=sio.loadmat('WB_test100.mat')['WB_test100']
data1=WB[0:100]
X_train=data1/1.
X_test=data1/1.
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

print np.min(X_train), np.max(X_train)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D,Lambda
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
import scipy.io as sio

batch_size = 1
original_dim = 100*100
latent_dim = 30
nb_epoch = 100

x = Input(shape=(1, 100, 100))
#h = Flatten()(x)
#x = Input(batch_shape=(batch_size, original_dim))
#h = Reshape((1,28,28))(x)
encode_h1=Convolution2D(24, 6, 6, activation='relu', border_mode='same')
encode_h2=MaxPooling2D((2, 2), border_mode='same')
encode_h3=Convolution2D(40, 9, 9, activation='relu', border_mode='same')
encode_h4=MaxPooling2D((2, 2), border_mode='same')
encode_h5= Convolution2D(144, 9, 9, activation='relu', border_mode='same')
# encode_h6=MaxPooling2D((2, 2), border_mode='same')
encode_h6=Flatten()
encode_h7=Dense(1000,activation='relu')

h = encode_h1(x)
h = encode_h2(h)
h = encode_h3(h)
h = encode_h4(h)
h = encode_h5(h)
h = encode_h6(h)
h = encode_h7(h)

# h = Flatten()(h)

z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)
encoder=Model(x,z_mean)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later

decoder_h0 = Dense(1000, activation='relu')
decoder_h1 = Dense(144*25*25, activation='relu')
decoder_h2 = Reshape((144,25,25))
decoder_h3 = Convolution2D(144, 9, 9, activation='relu', border_mode='same')
decoder_h4 = UpSampling2D((2, 2))
decoder_h5 = Convolution2D(40, 9, 9, activation='relu', border_mode='same')
decoder_h6 = UpSampling2D((2, 2))
decoder_h7 = Convolution2D(24, 6, 6, activation='relu',border_mode='same')
# decoder_h8 = UpSampling2D((2, 2))
decoder_h9 = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')

#g_input = Input(shape=[2])
H = decoder_h0(z)
H = decoder_h1(H)
H = decoder_h2(H)
H = decoder_h3(H)
H = decoder_h4(H)
H = decoder_h5(H)
H = decoder_h6(H)
H = decoder_h7(H)
# H = decoder_h8(H)
g_V = decoder_h9(H)

#generator = Model(z, g_V)

def vae_loss(x, g_V):
    xent_loss = original_dim * K.mean(objectives.binary_crossentropy(x, g_V))
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + kl_loss

vae = Model(x, g_V)
#keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
vae.compile(optimizer='rmsprop', loss=vae_loss)

vae.fit(X_train, X_train,
        shuffle=True,
        nb_epoch=nb_epoch,
        batch_size=batch_size,
        validation_data=(X_test, X_test))