#Where we put the data processing code#
import pandas as pd
from sklearn.model_selection import train_test_split

class Process:
  def __init__(self, path):
    self.path=path
    self.dataset=None
    self.X=None
    self.y=None
    self.X_train = None
    self.X_test = None
    self.y_train = None
    self.y_test = None
    
  def read(self):
    self.dataset=pd.read_csv(self.path)
    return print("read success")
    
  def clean(self):
    for column in self.dataset.columns:
        if column == target:
            continue

        unique_count = self.dataset[column].nunique()
        unique_ratio = unique_count / rows

        if unique_ratio >= threshold:
            self.dataset = self.dataset.drop(column, axis=1)
            print(f"Dropped {column}")
          
    return print("filter success")
    
  def allocate(self,ans):
    self.X = self.dataset.drop(ans, axis=1,)
    self.y = self.df[ans]
    return print("allocated X and y")
  
  def split(self, ratio):
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y,test_size=ratio,random_state=42)
    return print("Dataset split for testing and training")

  def process(self,target,ratio):
    print("reading Dataset")
    self.read()
    print("filtering Dataset for unuseable features")
    self.filter()
    print("allocating X and y")
    self.allocate(target)
    print("splitting for testing and training")
    self.split(ratio)
