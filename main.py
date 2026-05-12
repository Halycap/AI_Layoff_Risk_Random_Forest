from sklearn.ensemble import RandomForestClassifier
from Preprocess import Process
import os

#THE PROCESS BLOCK#
#Settings
path = "data/student_performance_updated_1000.csv"
feature_predict = "FinalGrade"
test_ratio = 0.2
random_split=42
random_train=42
tree_size=100

Pro=Process(path)
Pro.process(feature_predict,test_ratio,random_split)

#TRAINING BLOCK
tree=RandomForestClassifier(n_estimators=tree_size, random_state=random_train)

model.fit(Pro.X_train, Pro.y_train)

#OUTPUT BLOCK
predict= model.predict(Pro.X_test)

accuracy= model.accuracy(Pro.y_test,predict)

# Image Block for modeling
plt.figure(figsize=(20,10)) 
plot_tree(
  model.estimators_[0],
  feature_names=Pro.X.columns,
  class_names=[str(c) for c in model.classes_],
  filled=True,
  rounded=True,
  max_depth=3
)
plt.savefig("random_forest_tree.png", dpi=300, bbox_inches="tight)
plt.show()
