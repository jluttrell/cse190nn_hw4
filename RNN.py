import numpy as np
import random

num_chars = 256

class RNN:
  '''
  Character level recurrent neural network framework adapted from code from WildML blog
  (www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/)
  '''

  def __init__(self, num_hidden, temp):

    self.num_hidden = num_hidden
    self.temp = temp

    #input to hidden
    self.W_xh = np.random.uniform(-np.sqrt(1./num_chars), np.sqrt(1./num_chars), (num_hidden, num_chars))
    #hidden to output
    self.W_ho = np.random.uniform(-np.sqrt(1./num_hidden), np.sqrt(1./num_hidden), (num_chars, num_hidden))
    #recurrent connection in the hidden
    self.W_hh = np.random.uniform(-np.sqrt(1./num_hidden), np.sqrt(1./num_hidden), (num_hidden, num_hidden))

  def loss():
    return -1

  def forwardProp(self, x):
    T = len(x)

    hidden = np.zeros((T+1, self.num_hidden))
    hidden[-1] = np.zeros(self.num_hidden)

    out = np.zeros((T, num_chars))

    for t in np.arange(T):
      hidden[t] = np.tanh(self.W_xh[:,x[t]] + np.dot(self.W_hh, hidden[t-1]))
      out[t] = softmax(np.dot(self.W_ho, hidden[t]), self.temp)
    return out, hidden

  def bptt(self, x, y):
    T = len(y)
    
    o, h = self.forwardProp(x)

    dLdW_xh = np.zeros(self.W_xh.shape)
    dLdW_ho = np.zeros(self.W_ho.shape)
    dLdW_hh = np.zeros(self.W_hh.shape)

    delta_out = o
    delta_out[np.arange(len(y)),y] -= 1

    for t in np.arange(T)[::-1]:
      dLdW_ho += np.outer(delta_out[t], h[t].T)
      delta_t = np.dot(self.W_ho.T, delta_out[t]) * (1 - (h[t] ** 2))

      for step in np.arange(max(0, t), t+1)[::-1]:
        dLdW_hh += np.outer(delta_t, h[step-1])
        dLdW_xh[:,x[step]] += delta_t

        delta_t = np.dot(self.W_hh.T, delta_t) * (1 - (h[step-1] ** 2))
    
    return dLdW_xh, dLdW_ho, dLdW_hh

  def predict(self, x):
    out, hidden_states = self.forwardProp(x)
    #ascii_number = np.argmax(out, axis=1)
    ascii_number = np.argmax(out[-1])
    #return ascii_number[0]
    return ascii_number

  def sgd_step(self, x, y, lr):
    dLdW_xh, dLdW_ho, dLdW_hh = self.bptt(x,y)
    self.W_xh -= lr*dLdW_xh
    self.W_ho -= lr*dLdW_ho
    self.W_hh -= lr*dLdW_hh

### End RNN

def train(model, x, lr, sequence_len, num_epochs, print_freq):
  for epoch in range(num_epochs):
    i = 0
    if epoch % print_freq == 0:
      print 'epoch ', epoch 
    while (i+1+sequence_len) < len(x):
      model.sgd_step(x[i:i+sequence_len],x[i+1:i+1+sequence_len], lr)
      i += sequence_len

def softmax(a,temp):
  numer = np.exp(a/temp)
  out = numer / numer.sum()
  return out

def createX(filename):
  filesize = len(open(filename).read())
  print ('File has %d bytes' %filesize)
  f = open(filename)
  x = []
  c = f.read(1)
  while c is not None:
    if len(c) == 0:
      break
    x.append(ord(c))
    c = f.read(1)
  print ('Input contains %d characters' %len(x))
  if len(x) != filesize:
    print 'WARNING: Not all characters from file were read'
  return x

def generate(model, start, length):
  text = []
  text.append(start)

  for i in range(length-1):
    next_char = model.predict(text)
    text.append(next_char)
  return text

def main():

  ##### PARAMETERS #####
  filename = 'minutemysteries.txt'
  hidden_size = 150
  temp = 1
  learning_rate = 0.1
  sequence_len = 15
  num_epochs = 5
  print_freq = 1

  #start character of generated text
  start = ord('A')
  #start = random.randint(0,127)
  #how many characters to generate including start
  gen_length = 100

  ##### END PARAMETERS #####

  print 'Training...'
  print ('hidden_size = %d, learning_rate = %f, sequence_len = %d, num_epochs = %d' \
  x = createX(filename)
    %(hidden_size, learning_rate, sequence_len, num_epochs))

  net = RNN(hidden_size, temp)
  train(net, x, learning_rate, sequence_len, num_epochs, print_freq)

  print ('Generating text of length %d' %gen_length)
  gen = generate(net, start, gen_length)

  #convert ascii # to char
  gen = [str(unichr(x)) for x in gen]

  #join list of chars and print
  print 'Generated text: '
  print ''.join(gen)

  print "\ndone!"

if __name__ == '__main__':
  main()