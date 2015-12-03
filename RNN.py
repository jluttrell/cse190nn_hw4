from ReadFile import ReadFile
from Functions import *

class RNN:

  num_chars = 256

  def __init__(self, filename, num_hidden, seq_length, temp):
    self.text = ReadFile(filename)

    self.num_hidden = num_hidden
    self.seq_length = seq_length
    self.temp = temp

    #input to hidden
    self.W_xh = np.random.uniform(-np.sqrt(1./num_chars), np.sqrt(1./num_chars), (num_hidden, num_chars))
    #hidden to output
    self.W.ho = np.random.uniform(-np.sqrt(1./num_hidden), np.sqrt(1./num_hidden), (num_chars, num_hidden))
    #recurrent connection in the hidden
    self.W_hh = np.random.uniform(-np.sqrt(1./num_hidden), np.sqrt(1./num_hidden), (num_hidden, num_hidden))


  def forwardProp(self, x):
    return 0

  def bptt(self, output, softmax_output):
    dLdU = np.zeros(self.W_xh.shape)
    dLdV = np.zeros(self.W_ho.shape)
    dLdW = np.zeros(self.w_hh.shape)

    delta_out = output

  def predict(self, x):
    out, hidden_states = self.forwardProp(x)
    ascii_number = np.argmax(out, axis=1)
    return str(unichr(ascii_number))

  def train(self, readfile):
    return 0

def main():
  net = RNN('infile.txt', 10, 10, 1)
  net.train(f)
  net.test()

  print "done"

if __name__ == '__main__':
  main()

  # c = f.readChar()
  # while c is not None:
  #   i = f.createInputVector(c)
  #   c = f.readChar()


for t in range len(y):
  dLdU = np.zeros(self.W_xh.shape)
  dLdV = np.zeros(self.W_ho.shape)
  dLdW = np.zeros(self.w_hh.shape)

  delta_o = output
  dLdV = outer of delta_out and softmax output
  delta_t = W_ho dot delta_out * (1 - softmax_output**2)

  for step in range (t+1, max(0, t - seq_length)):
