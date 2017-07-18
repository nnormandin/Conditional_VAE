import numpy as np
from keras.layers import Input, Dense, Lambda
from keras.layers.merge import concatenate
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from scipy.misc import imsave


# load mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# convert y to one-hot, reshape x
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))

# select optimizer
optim = 'adam'

# dimension of latent space (batch size by latent dim)
m = 50
n_z = 50

# dimension of input (and label)
n_x = X_train.shape[1]
n_y = y_train.shape[1]

# nubmer of epochs
n_epoch = 10

##  ENCODER ##

# encoder inputs
X = Input(shape=(784, ))
cond = Input(shape=(n_y, ))

# merge pixel representation and label
inputs = concatenate([X, cond])

# dense ReLU layer to mu and sigma
h_q = Dense(512, activation='relu')(inputs)
mu = Dense(n_z, activation='linear')(h_q)
log_sigma = Dense(n_z, activation='linear')(h_q)

def sample_z(args):
    mu, log_sigma = args
    eps = K.random_normal(shape=(m, n_z), mean=0., stddev=1.)
    return mu + K.exp(log_sigma / 2) * eps


# Sampling latent space
z = Lambda(sample_z, output_shape = (n_z, ))([mu, log_sigma])

# merge latent space with label
z_cond = concatenate([z, cond])

##  DECODER  ##

# dense ReLU to sigmoid layers
decoder_hidden = Dense(512, activation='relu')
decoder_out = Dense(784, activation='sigmoid')
h_p = decoder_hidden(z_cond)
outputs = decoder_out(h_p)

# define cvae and encoder models
cvae = Model([X, cond], outputs)
encoder = Model([X, cond], mu)

# reuse decoder layers to define decoder separately
d_in = Input(shape=(n_z+n_y,))
d_h = decoder_hidden(d_in)
d_out = decoder_out(d_h)
decoder = Model(d_in, d_out)

# define loss (sum of reconstruction and KL divergence)
def vae_loss(y_true, y_pred):
    # E[log P(X|z)]
    recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=1)
    # D_KL(Q(z|X) || P(z|X))
    kl = 0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1)
    return recon + kl

def KL_loss(y_true, y_pred):
	return(0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1))

def recon_loss(y_true, y_pred):
	return(K.sum(K.binary_crossentropy(y_pred, y_true), axis=1))


# compile and fit
cvae.compile(optimizer=optim, loss=vae_loss, metrics = [KL_loss, recon_loss])
cvae_hist = cvae.fit([X_train, y_train], X_train, batch_size=m, epochs=n_epoch,
							validation_data = ([X_test, y_test], X_test),
							callbacks = [EarlyStopping(patience = 5)])

# this loop prints the one-hot decodings

#for i in range(n_z+n_y):
#	tmp = np.zeros((1,n_z+n_y))
#	tmp[0,i] = 1
#	generated = decoder.predict(tmp)
#	file_name = './img' + str(i) + '.jpg'
#	print(generated)
#	imsave(file_name, generated.reshape((28,28)))
#	sleep(0.5)

# this loop prints a transition through the number line

pic_num = 0
variations = 30 # rate of change; higher is slower
for j in range(n_z, n_z + n_y - 1):
	for k in range(variations):
		v = np.zeros((1, n_z+n_y))
		v[0, j] = 1 - (k/variations)
		v[0, j+1] = (k/variations)
		generated = decoder.predict(v)
		pic_idx = j - n_z + (k/variations)
		file_name = './transition_50/img{0:.3f}.jpg'.format(pic_idx)
		imsave(file_name, generated.reshape((28,28)))
		pic_num += 1
		
		
		
		