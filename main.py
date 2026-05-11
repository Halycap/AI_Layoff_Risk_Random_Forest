from sklearn.ensemble import RandomForestClassifier
from Preprocess import Process
import os

#THE PROCESS BLOCK#
path = ""
feature_predict = ""
training_ratio = ""

Pro=Process(path)
Pro.process(feature_predict,training_ratio)

#TRAINING BLOCK
tree=RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(Pro.X_train, Pro.y_train)

predict= model.predict(Pro.X_test)

accuracy= model.accuracy(Pro.y_test,predict)
#OUTPUT BLOCK

# Image Block
plt.figure(figsize=(20,10)) 
plot_tree(
  model.estimators_[0],
  feature_names=Pro.X.columns,
  class_names=[str(c) for c in model.classes_],
  filled=True,
  rounded=True,
  max_depth=3
)
