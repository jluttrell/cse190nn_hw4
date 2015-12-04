from ReadFile import ReadFile
import numpy as np

class RNN:

  num_chars = 256

  def __init__(self, num_hidden, seq_length, temp):

    self.num_hidden = num_hidden
    self.seq_length = seq_length
    self.temp = temp

    #input to hidden
    self.W_xh = np.random.uniform(-np.sqrt(1./num_chars), np.sqrt(1./num_chars), (num_hidden, num_chars))
    #hidden to output
    self.W.ho = np.random.uniform(-np.sqrt(1./num_hidden), np.sqrt(1./num_hidden), (num_chars, num_hidden))
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
      out[t] = softmax(np.dot(self.W_ho, s[t]), self.temp)
    return out, hidden

  def bptt(self, output, hidden, x, y):
    T = len(y)
    dLdW_xh = np.zeros(self.W_xh.shape)
    dLdW_ho = np.zeros(self.W_ho.shape)
    dLdW_hh = np.zeros(self.w_hh.shape)

    delta_out = output - y

    for t in np.arange(T)[::-1]:
      dLdW_ho += np.outer(delta_out[t], hidden[t].T)
      delta_t = np.dot(W_ho.T, delta_out[t]) * (1 - (hidden[t] ** 2))

      for step in np.arange(max(0, t- self.seq_length), t+1)[::-1]:
        dLdW_hh += np.outer(delta_t, hidden[s-1])
        dLdW_xh[:,x[step]] += delta_t

        delta_t = np.dot(W_hh.T, delta_t) * (1 - (hidden[step-1] ** 2))
    
    return dLdW_xh, dLdW_ho, dLdW_hh

  def predict(self, x):
    out, hidden_states = self.forwardProp(x)
    ascii_number = np.argmax(out, axis=1)
    return str(unichr(ascii_number))

  def train(self, readfile):
    return -1

def softmax(a,temp):
  numer = np.exp(a/temp)
  out = numer / numer.sum()
  return out

def main():
  text = ReadFile('infile.txt')
  x = text.createX()
  hidden_size = 10
  sequence_len = 10
  temp = 1
  net = RNN(hidden_size, sequence_len, temp)

  print "done"

if __name__ == '__main__':
  main()

  # c = f.readChar()
  # while c is not None:
  #   i = f.createInputVector(c)
  #   c = f.readChar()