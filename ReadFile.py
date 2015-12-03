class ReadFile:

  def __init__(self, filename):
    self.file = open(filename)

  def readChar(self):
    c = self.file.read(1)
    
    if not c:
      return None
    else:
      return ord(c)
      
def main():
  f = ReadFile('infile.txt')
  c = f.readChar()
  while c is not None:
    print f.createInputVector(c)
    c = f.readChar()
  print "done"

#if __name__ == "__main__":
    #main()