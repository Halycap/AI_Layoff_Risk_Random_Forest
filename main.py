from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from Preprocess import Process
from additional_code_for_good_graphs import importance_graph
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from pathlib import Path

#Settings
BASE_DIR = Path(__file__).resolve().parent
#for testing
#step 1, change the path name with the dataset path
path = BASE_DIR / "data" / "ai-impact-jobs-layoff-risk-dataset.csv"
#step 2, change the feature
feature_predict = "Layoff_Risk"
#step 3, mess with the settings
missing_limit=0.9 #clears all rows that are missing values
correlation=0.5 #filters all that correlates below this range
test_ratio = 0.2 #splits the data to have this percent of test values
random_split=42 #random seed for split 
random_train=42 #random seed for train
tree_size=100 #how big the tree is
result_mode="present" #test or present, test doesnt run the tree plot and present generates the tree models

#DATA PROCESSING BLOCK
Pro = Process(path) #Calls the custom class i made built for data processing
Pro.process(feature_predict,test_ratio,random_split,correlation,missing_limit) #yes its a lot. more settings = more stuff to write
data_type = Pro.data_type_identifier(feature_predict) #for regressor or classifier model logic

#TRAINING BLOCK
if data_type == "category":
    tree = RandomForestClassifier(n_estimators=tree_size, random_state=random_train)
    print("classifier")
else:
    tree = RandomForestRegressor(n_estimators=tree_size, random_state=random_train)
    print("regressor")

tree.fit(Pro.X_train, Pro.y_train)

#OUTPUT BLOCK
#Prediction test. WIll kill this dual model system
y_pred = tree.predict(Pro.X_test)
if data_type == "category":
    accuracy = accuracy_score(Pro.y_test, y_pred)
    print("Accuracy:", accuracy)
else:
    mae = mean_absolute_error(Pro.y_test, y_pred)
    r2 = r2_score(Pro.y_test, y_pred)

    print("Mean Absolute Error:", mae)
    print("R² Score:", r2)

# Image Block for modeling
if data_type == "category":
    class_names = [str(c) for c in tree.classes_]
else:
    class_names = None

plt.figure(figsize=(20,10)) 

if result_mode="present":
    for i in range(5):
        plt.figure(figsize=(20, 10))
    
        plot_tree(
            tree.estimators_[i],
            feature_names=Pro.X.columns,
            class_names=class_names,
            filled=True,
            rounded=True,
            max_depth=3
        )
    
        plt.savefig(f"random_forest_tree_{i}.png", dpi=300, bbox_inches="tight")
        plt.close()
    importance_graph(tree,Pro)



