import sklearn.ensemble import RandomForestClassifier
from Preprocess import Process

target="x"
#THE PROCESS BLOCK#
Pro=Process("data/AI_Student_Life_Pakistan_2026.csv")
Pro.read()
X,y=Pro.process(target)
X_train,X_test,Y_train,Y_test=pro.split(0.2)
