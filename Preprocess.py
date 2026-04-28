#Where we put the data processing code#
import pandas as pd
from sklearn.model_selection import train_test_split:

class Process
  def __init__(self, path):
    self.path=path
    self.dataset=None
    
  def read(self):
    self.dataset=pd.read_csv(self.path)
    return self.dataset

  def process(self):
    #TO DO#
  
  def split(X,Y,ratio):
    return train_test_split(X,Y,test_size=ratio,random_state=42)
