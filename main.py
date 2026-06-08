from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from Preprocess import Process
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from pathlib import Path


#THE PROCESS BLOCK#
#Settings
BASE_DIR = Path(__file__).resolve().parent
path = BASE_DIR / "data" / "student_performance_updated_1000.csv"
feature_predict = "FinalGrade"
test_ratio = 0.2
random_split=42
random_train=42
tree_size=100

Pro = Process(path)
Pro.process(feature_predict,test_ratio,random_split)

#TRAINING BLOCK
if Pro.data_type_identifier(feature_predict) == "category":
    tree = RandomForestClassifier(n_estimators=tree_size, random_state=random_train)
    print("classifier")
else:
    tree = RandomForestRegressor(n_estimators=tree_size, random_state=random_train)
    print("regressor")

tree.fit(Pro.X_train, Pro.y_train)

#OUTPUT BLOCK
#Prediction test. WIll kill this dual model system
y_pred = tree.predict(Pro.X_test)
if Pro.data_type_identifier(feature_predict) == "category":
    accuracy = accuracy_score(Pro.y_test, y_pred)
    print("Accuracy:", accuracy)
else:
    mae = mean_absolute_error(Pro.y_test, y_pred)
    r2 = r2_score(Pro.y_test, y_pred)

    print("Mean Absolute Error:", mae)
    print("R² Score:", r2)

# Image Block for modeling
# Image Block for modeling
if Pro.data_type_identifier(feature_predict) == "category":
    class_names = [str(c) for c in tree.classes_]
else:
    class_names = None

plt.figure(figsize=(20,10)) 

plot_tree(
    tree.estimators_[0],
    feature_names=Pro.X.columns,
    class_names=class_names,
    filled=True,
    rounded=True,
    max_depth=3
)

plt.savefig("random_forest_tree.png", dpi=300, bbox_inches="tight")
plt.show()
