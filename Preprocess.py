#Where we put the data processing code#
import pandas as pd
from sklearn.model_selection import train_test_split
from pandas.api.types import is_numeric_dtype

class Process:
  def __init__(self, path):
    self.path=path
    self.dataset=pd.read_csv(self.path)
    self.X=None
    self.y=None
    self.X_train = None
    self.X_test = None
    self.y_train = None
    self.y_test = None
  #This one is utils. but i dont wanna make a utils folder 0v0
  def data_type_identifier(self, target, unique_threshold=10):
    y = self.dataset[target]
    unique_count = y.nunique()
    is_numeric = is_numeric_dtype(y)

    if not is_numeric:
        return "classifier", False

    if unique_count <= unique_threshold:
        return "classifier", True

    return "regressor", True
    
  def clean_column(self,target,threshold=0.8):
    rows = len(self.dataset)
    for column in self.dataset.columns:
      if column == target:
          continue
      
      unique_count = self.dataset[column].nunique()
      unique_ratio = unique_count / rows
     
      if unique_ratio >= threshold:
          self.dataset = self.dataset.drop(column, axis=1)
          print(f"Dropped {column}")
       
    return print("column filter success")
    
  def clean_row(self,target,missing_limit=0.5):
    self.dataset = self.dataset.dropna(subset=[target])
    
    feature_columns = self.dataset.columns.drop(target)
    
    missing_ratio = self.dataset[feature_columns].isna().mean(axis=1)
    
    self.dataset = self.dataset[missing_ratio <= missing_limit]
    
    self.dataset = self.dataset.drop_duplicates()
    return print("row filter success")

  def clean(self, target):
    self.clean_column(target)
    self.clean_row(target)

  def fill(self,target):
    feature_columns = self.dataset.columns.drop(target)
    for column in feature_columns:
        if self.dataset[column].dtype == "object":
            most_common = self.dataset[column].mode()[0]
            self.dataset[column] = self.dataset[column].fillna(most_common)
        else:
            median_value = self.dataset[column].median()
            self.dataset[column] = self.dataset[column].fillna(median_value)
        return print("missing entries filled with median")
    
  def allocate(self,ans):
    self.X = self.dataset.drop(ans, axis=1,)

    # converts text into numbers. Model, why do u need numbers.
    self.X = pd.get_dummies(self.X)
    
    self.y = self.dataset[ans]
    return print("allocated X and y")
  
  def split(self, ratio,random):
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y,test_size=ratio,random_state=random)
    return print("Dataset split for testing and training")

  def process(self,target,ratio,random=42):
    print("filtering Dataset for unuseable features")
    self.clean(target)

    print("attempting to fill missing data entries")
    self.fill(target)

    print("filtering weakly correlated numeric features")
    self.correlation_filter(target)

    print("allocating X and y")
    self.allocate(target)

    print("splitting for testing and training")
    self.split(ratio,random)

