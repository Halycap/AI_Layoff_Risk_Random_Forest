from sklearn.ensemble import RandomForestClassifier
from Preprocess import Process
import os

#THE PROCESS BLOCK#
default_path = "data/AI_Student_Life_Pakistan_2026.csv"
feature_predict = ""
training_ratio = ""

Pro=Process(path)
Pro.process(feature_predict,training_ratio)

#TRAINING BLOCK


#OUTPUT BLOCK
