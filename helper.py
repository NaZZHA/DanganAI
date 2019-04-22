import numpy as np

def random_normal(size):
	return np.random.normal(size=size)

def clean_filename(fn):
	new_filename = []
	for i in fn:
		if i == ' ' or i =='%':
			i = '_'
			continue
		if i == '(' or i == ')':
			continue
		new_filename.append(i)
	return ''.join(new_filename)

softmax = lambda x : np.exp(x) / np.sum(np.exp(x), axis=0) 
relu = lambda x: x * (x > 0)
sigmoid = lambda x : 1 / (1 + np.exp(-x))
reciprocal = lambda x : 1 / (x + 1)