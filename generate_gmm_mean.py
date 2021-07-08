import numpy as np 
import scipy.io as sio

np.random.seed(0)

num_train = 50
num_val = 25
num_test = 25

train_mean = np.random.rand(num_train, 2)
val_mean = np.random.rand(num_val, 2)
test_mean = np.random.rand(num_test, 2)

sio.savemat('gmm_mean', {'train_mean': train_mean, 'val_mean': val_mean, 'test_mean': test_mean})