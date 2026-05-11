#Where we put the data processing code#
import pandas as pd
from sklearn.model_selection import train_test_split:

class Process
  def __init__(self, path):
    self.path=path
    self.dataset=None
    self.X=None
    self.Y=None
    self.X_train = None
    self.X_test = None
    self.y_train = None
    self.y_test = None
    
  def read(self):
    self.dataset=pd.read_csv(self.path)
    return print("read success")
    
  def filter(self):
    self.dataset
    return print("filter success")
    
  def allocate(self,ans):
    self.X = self.df.drop(ans, axis=1,)
    self.y = self.df[ans]
    return print("allocated X and y")
  
  def split(self, ratio):
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y,test_size=ratio,random_state=42)
    return print("Dataset split for testing and training")

  def process(self,y,ratio):
    print("reading Dataset")
    read(self)
    print("filtering Dataset for unuseable features")
    filter(Self)
    print("allocating X and y")
    allocate(self,y)
    print("splitting for testing and training")
    split(self,ratio)
