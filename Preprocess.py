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
  def data_type_identifier(self, target):
    y = self.dataset[target]

    if not is_numeric_dtype(y):
        return "category"
    return "value"
    
  def clean_column(self,target,threshold=0.8):
    #clean columns cleans columns that has too many unique entries(treshold) and takes a treshold value to cut off certain columms
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
    
  def clean_row(self,target,missing_limit=0.1):
    #cleans rows that have missing_limit amount of missing entries
    self.dataset = self.dataset.dropna(subset=[target])
    
    feature_columns = self.dataset.columns.drop(target)
    
    missing_ratio = self.dataset[feature_columns].isna().mean(axis=1)
    
    self.dataset = self.dataset[missing_ratio <= missing_limit]
    
    self.dataset = self.dataset.drop_duplicates()
    return print("row filter success")

  def clean(self, target,missing_limit):
    # combines cleaning rows and columns
    self.clean_column(target)
    self.clean_row(target,missing_limit)

  def fill(self, target):
    # if relevant, fills up remaining missing entries to median
    feature_columns = self.dataset.columns.drop(target)

    for column in feature_columns:
        if is_numeric_dtype(self.dataset[column]):
            median_value = self.dataset[column].median()
            self.dataset[column] = self.dataset[column].fillna(median_value)
        else:
            most_common = self.dataset[column].mode()

            if not most_common.empty:
                self.dataset[column] = self.dataset[column].fillna(most_common[0])

    return print("missing entries filled")
  
  def correlate(self, target, threshold):
    # removes columns that dont correlate with a target
    # it cannot remove columns if the target is not a value or the columns tested are categories
    target_type = self.data_type_identifier(target)

    # Only use correlation filter for regression/value targets
    if target_type != "value":
        print("correlation filter skipped for target")
        return

    numeric_columns = []

    for column in self.dataset.columns:
        if self.data_type_identifier(column) == "value":
            numeric_columns.append(column)

    correlations = self.dataset[numeric_columns].corr()[target].abs()

    weak_columns = correlations[correlations < threshold].index

    # never drop the target value, i learnt my lesson
    weak_columns = weak_columns.drop(target, errors="ignore")

    self.dataset = self.dataset.drop(columns=weak_columns)

    print(f"Dropped columns: {list(weak_columns)}")
    
  def allocate(self,ans):
    self.X = self.dataset.drop(ans, axis=1,)

    # converts text into numbers. Model, why do u need numbers.
    self.X = pd.get_dummies(self.X)
    
    self.y = self.dataset[ans]
    return print("allocated X and y")
  
  def split(self, ratio,random):
    #train and test values now being allocated
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y,test_size=ratio,random_state=random)
    return print("Dataset split for testing and training")

  def process(self,target,ratio,random=42,correlation=0.1,missing_limit=0.5):
    # the mega function. this way only one function is called outside instead of 5 lines.
    print("filtering Dataset for unuseable features")
    self.clean(target,missing_limit)

    print("attempting to fill missing data entries")
    self.fill(target)

    print("correlation filter")
    self.correlate(target,correlation)

    print("allocating X and y")
    self.allocate(target)

    print("splitting for testing and training")
    self.split(ratio,random)

