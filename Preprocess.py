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

  def features(self):
    print("Features available:")
    for column in self.df.columns:
      print(column,"/")

  def ask_target(self):
    target_column = input("Feature Predicting: ")
    return target_column

  def process(self,ans):
    self.X = self.df.drop(ans, axis=1,)
    self.y = self.df[ans]
    return print("process success")
  
  def split(self, ratio):
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y,test_size=ratio,random_state=42)
