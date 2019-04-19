import numpy as np

def random_normal(size):
	return np.random.normal(size=size)

softmax = lambda x : np.exp(x) / np.sum(np.exp(x), axis=0) 
relu = lambda x: x * (x > 0)
sigmoid = lambda x : 1 / (1 + np.exp(-x))