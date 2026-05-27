This project was created for the Introduction to AI course.

The goal of this project is to use a Decision Tree machine learning model to predict student performance based on different factors from a dataset found on Kaggle.
If you need to analyze educational data and predict student outcomes based on lifestyle factors, this project provides a working template. It takes a Kaggle dataset containing student metrics (study habits, attendance, sleep, grades) and trains a machine learning model to predict final academic success with a target accuracy score.
This project is built completely on open-source Python libraries (Scikit-learn, Pandas, NumPy, and Matplotlib). You can copy this code, download the public Kaggle dataset, and run the entire machine learning pipeline locally on your computer by following the setup steps below.

The dataset contains information about students such as:
- study habits
- attendance
- sleep hours
- previous grades
- lifestyle information
- and other features related to student performance

### Dataset Name
Student Performance Dataset

You can download the dataset here:

https://kaggle.com

This project uses:
- Python
- Pandas
- Scikit-learn
- Matplotlib

## What you need first

Thanks for checking out our project. This guide will walk you through getting the code running on your computer. 

Before we start, you just need three things:
1. Python (version 3.8 or newer) - this is what our code is written in
2. Git - just a small tool to grab the code from GitHub
3. A code editor (not required but nice to have) - try VS Code if you want

# Getting Python

1.  Go here: https://www.python.org/downloads/
2.  Hit that big yellow Download Python button
3.  Open the file you just downloaded
4.  Super important: On the first screen, tick the box that says "Add Python to PATH". Do not skip this.
5.  Click "Install Now"
6.  Click "Close" when it's done

## Grabbing the code from GitHub

1.  Go to our project page: https://github.com/Halycap/AI_Student_life_DT
2.  Find and click the green "<> Code" button
3.  From the little menu, click "Download ZIP"
4.  Go to your Downloads folder, find the zip file, and right-click it
5.  Choose "Extract All" and put the folder somewhere easy to find - your "Desktop" would work great

## Opening up the terminal

This is just where you'll type the commands to run everything.
- On Windows: Press the Windows key, type cmd, hit Enter
- On Mac: Press Command + Space, type Terminal, hit Enter

## Finding your way to the project folder

Now we tell the terminal where our code lives.

First, type this and hit Enter:
"cd Desktop" 

(If you put the folder somewhere else like Documents, type "cd Documents" instead)

Then type this and hit Enter:

"cd AI_Student_life_DT-main"

(The folder name might be a little different. If you're not sure, type "dir" on Windows or "ls" on Mac to see what folders are there.)

Once you've done that, you're inside the project folder.

## Installing the required libraries (do this ONCE before running)

Our code needs three extra pieces to work. You only need to install them once. Type this command and hit Enter:

"pip install scikit-learn pandas matplotlib"


Here's what each one does for our project:

- scikit-learn: This is the main machine learning library. It gives us the Random Forest model (RandomForestClassifier), the function to split data into training/testing (train_test_split), the accuracy score calculator (accuracy_score), and the tree visualizer (plot_tree)
- pandas: This reads our CSV file (student_performance_updated_1000.csv) and handles all the data cleaning - dropping empty rows, filling missing values, removing duplicate entries. Basically, it manages our entire dataset
- matplotlib: This draws the decision tree picture and saves it as "random_forest_tree.png" so you can actually see what the AI learned-

Wait for the installation to finish. Once it's done, you're ready to run the code.

## Actually running the code

Type this and hit Enter:

python main.py

You'll see messages pop up. Here ia what the program is doing behind the scenes:
1. Reading the student data from the CSV file
2. Cleaning it up (real data is messy - it removes bad rows, fills in missing numbers)
3. Converting text categories into numbers the AI can understand
4. Training an AI model (Random Forest) to predict final grades
5. Showing you how well it did with an accuracy score
6. Saving a picture of the decision tree

### What should show up
You will see things like "read success", "column filter success", "allocated X and y", and "dataset split for testing and training" in the terminal.

At the end it will print out a number, that's the accuracy of our model. Something like 0.85 means it was right 85% of the time when predicting student grades.

Also, check the project folder afterward. There should be a picture file called random_forest_tree.png. Double-click it to see a visual map of what the AI learned. It shows the questions the model asks to make its predictions.


Here's what to do if things go wrong.

Problem: "python" is not recognized as an internal or external command
    - Fix: Python didn't install right. Go back to the Python installation step and make sure you checked "Add Python to PATH". Then close your terminal and open it again.
Problem: ModuleNotFoundError: No module named "sklearn"
    - Fix: scikit-learn is missing. Run this command:
        pip install scikit-learn
Problem: ModuleNotFoundError: No module named "pandas"
    - Fix: pandas is missing. Run this command:
        pip install pandas
Problem: ModuleNotFoundError: No module named "matplotlib"
    - Fix: matplotlib is missing. Run this command:
        pip install matplotlib
Problem: File not found error (says something like "no such file or directory")
     Fix: You're probably in the wrong folder. Make sure your terminal path ends with AI_Student_life_DT-main. Type dir (Windows) or ls (Mac) and you should see main.py and Preprocess.py and a folder called data. If you don't see them, go back to the navigation step.
Problem: Something about KeyError: "FinalGrade"
    - Fix: The dataset file might be wrong. Make sure the folder called data is inside your project folder, and inside it there's a file called student_performance_updated_1000.csv. Also check that the column name is exactly FinalGrade (capital F, capital G)

That would be everything!
Thank you for taking a look at our project!
