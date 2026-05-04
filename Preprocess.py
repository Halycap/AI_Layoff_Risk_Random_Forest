#Where we put the data processing code#
import pandas as pd
from sklearn.model_selection import train_test_split:

class Process
  def __init__(self, path):
    self.path=path
    self.dataset=None
    self.X=None
    self.y=None
    
  def read(self):
    self.dataset=pd.read_csv(self.path)
    return self.dataset

  def features(self):
    print("Features available")
    for column in self.df.columns:
      print(", ", column)

  def ask_target(self):
    target_column = input("Feature Predicting: ")
    return target_column

  def process(self,ans):
      self.X = self.df.drop(ans, axis=1,)
      self.y = self.df[ans]
  
  def split(X,Y,ratio):
    return train_test_split(X,Y,test_size=ratio,random_state=42)
