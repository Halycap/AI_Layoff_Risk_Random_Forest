import sklearn.ensemble import RandomForestClassifier
from Preprocess import Process

#THE PROCESS BLOCK#
#Loads the class and inputs the path#
Pro=Process("data/AI_Student_Life_Pakistan_2026.csv")
#Reads said file to process#
Pro.read()
#Displays the features and asks user to input a feature for prediction#
Pro.features()
target=ask_target()
#extracts X and y as well as split it to prepare to put in scikit
X,y=Pro.process(target)
X_train,X_test,Y_train,Y_test=pro.split(X,y,0.2)

#training
