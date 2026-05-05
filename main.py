from sklearn.ensemble import RandomForestClassifier
from Preprocess import Process
import os

#THE PROCESS BLOCK#
def dataprocess():
  default_path = "data/AI_Student_Life_Pakistan_2026.csv"
  
  file_name = input("CSV file name, or press Enter for default: ")
  
  if file_name == "":
    path = default_path
  else:
    path = os.path.join("data/", file_name)
  
  if not os.path.exists(path):
    print("File not found. Using default dataset.")
    path = default_path
    
  #Loads the class and inputs the path#
  Pro=Process(path)
  #Reads said file to process#
  Pro.read()
  #Displays the features and asks user to input a feature for prediction#
  Pro.features()
  target=Pro.ask_target()
  #extracts X and y as well as split it to prepare to put in scikit
  Pro.process(target)
  Pro.split(0.2)
  
  return 0

#training
