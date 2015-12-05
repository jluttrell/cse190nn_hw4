import numpy as np
import random
import time

num_chars = 256

class RNN:
  '''
  Character level recurrent neural network framework adapted from code from WildML blog
  (www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/)
  '''

  def __init__(self, num_hidden):

    self.num_hidden = num_hidden

    #input to hidden
    self.W_xh = np.random.uniform(-np.sqrt(1./num_chars), np.sqrt(1./num_chars), (num_hidden, num_chars))
    #hidden to output
    self.W_ho = np.random.uniform(-np.sqrt(1./num_hidden), np.sqrt(1./num_hidden), (num_chars, num_hidden))
    #recurrent connection in the hidden
    self.W_hh = np.random.uniform(-np.sqrt(1./num_hidden), np.sqrt(1./num_hidden), (num_hidden, num_hidden))

  def loss(self,x):
    N = len(x)
    x = x[0:len(x)-1]
    y = x[1:len(x)]
    o, h = self.forwardProp(x)
    correct_word_predictions = o[np.arange(len(y)), y]
    L = -1 *np.sum(np.log(correct_word_predictions))
    return L/N

  def forwardProp(self, x, temp=1):
    T = len(x)

    hidden = np.zeros((T+1, self.num_hidden))
    hidden[-1] = np.zeros(self.num_hidden)

    out = np.zeros((T, num_chars))

    for t in np.arange(T):
      hidden[t] = np.tanh(self.W_xh[:,x[t]] + np.dot(self.W_hh, hidden[t-1]))
      out[t] = softmax(np.dot(self.W_ho, hidden[t]), temp)
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

  def predict(self, x, temp=1):
    out, hidden_states = self.forwardProp(x, temp)
    #ascii_number = np.argmax(out, axis=1)
    ascii_number = np.argmax(out[-1])
    #return ascii_number[0]
    return ascii_number

  def sgd_step(self, x, y, lr):
    dLdW_xh, dLdW_ho, dLdW_hh = self.bptt(x,y)

    self.W_xh -= lr*np.clip(dLdW_xh, -5, 5)
    self.W_ho -= lr*np.clip(dLdW_ho, -5, 5)
    self.W_hh -= lr*np.clip(dLdW_hh, -5, 5)


### End RNN

def train(model, x, lr, sequence_len, num_epochs, print_freq):
  text = []
  for epoch in range(num_epochs):
    i = 0
    if epoch % print_freq == 0:
      print time.strftime("%Y-%m-%d %H:%M:%S"),
      #loss = model.loss(x[:1000])
      text.append(generateBii(model, 25, 1))
      loss = 1
      print ('\tepoch #%d: \tloss = %f' %(epoch, loss))
    while (i+1+sequence_len) < len(x):
      model.sgd_step(x[i:i+sequence_len],x[i+1:i+1+sequence_len], lr)
      i += sequence_len
  return text

def softmax(a,temp):
  numer = np.exp(a/temp)
  out = numer / numer.sum()
  return out

def createX(filename):
  filesize = len(open(filename).read())
  print ('\nFile has %d bytes' %filesize)

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
  unique = len(set(x))
  print ('There are %d unique characters\n' %unique)
  return x

def generate(model, start, length, temp, sequence_len):
  text = []
  text.append(start)

  for i in range(length-1):
    beg = max(0, len(text)-sequence_len)
    end = len(text)
    next_char = model.predict(text[beg:end], temp)
    text.append(next_char)
  return text

def generateBii(model, length, temp):
  f = open('minutemysteries.txt')
  gen = []
  text = []
  c = f.read(1)
  for i in range(1000):
    if len(c) == 0:
      break
    text.append(ord(c))
    c = f.read(1)

  for i in range(length-1):
    next_char = model.predict(text, temp)
    text.append(next_char)
    gen.append(next_char)
  return gen

def main():

  ########## PARAMETERS ##########

  filename = 'minutemysteries.txt'
  hidden_size = 100
  temp = 1
  learning_rate = 0.01
  sequence_len = 50
  num_epochs = 25
  print_freq = 5
  temp = 1

  #start character of generated text
  start = ord('P')
  #start = random.randint(32,58)

  #how many characters to generate including start
  gen_length = 20

  ########## END PARAMETERS ##########

  x = createX(filename)

  print 'Training...'
  print ('hidden_size = %d, learning_rate = %f, sequence_len = %d, num_epochs = %d\n' \
    %(hidden_size, learning_rate, sequence_len, num_epochs))

  net = RNN(hidden_size)
  t = train(net, x, learning_rate, sequence_len, num_epochs, print_freq)

  for i in range(len(t)):
    print i
    sent = [str(chr(x)) for x in t[i]]
    print ''.join(sent)

  print ('\nGenerating text of length %d' %gen_length)

  gen = generate(net, start, gen_length, temp, sequence_len)

  #convert ascii # to char
  gen = [str(chr(x)) for x in gen]

  #join list of chars and print
  print 'Generated text: '
  print ''.join(gen)

  print "\ndone!"

if __name__ == '__main__':
  main()