import numpy as np

def softmax(z):
  a = np.array(z)
  numer = np.exp(a)
  out = numer / numer.sum()
  return out

def sigmoid(a):
  return 1 / (1 + np.exp(-a))

def sigmoid_prime(a):
  return np.multiply(sigmoid(a),(1 - sigmoid(a)))

def tanh(a):
  return np.tanh(a)

def tanh_prime(a):
  return [(1 - np.tanh(x))**2 for x in a]

def relu(a):
  return np.maximum(a,0)

def relu_prime(a):
  a[a > 0] = 1
  a[a < 0] = 0
  return a

def activation(a, funct):
  if funct == "sigmoid":
    return sigmoid(a)
  elif funct == "tanh":
    return tanh(a)
  else:
    return relu(a)

def activation_prime(a, funct):
  if funct == "sigmoid":
    return sigmoid_prime(a)
  elif funct == "tanh":
    return tanh_prime(a)
  else:
    return relu_prime(a)