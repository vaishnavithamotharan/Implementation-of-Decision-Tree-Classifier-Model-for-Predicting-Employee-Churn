# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load & Explore Dataset — Read the Employee CSV file and inspect the data using head(), info(), null checks, and target class distribution.
2.Preprocess Data — Apply Label Encoding on the categorical "salary" column to convert it into numerical format.
3.Split Data — Separate features (satisfaction level, evaluation, projects, hours, etc.) and target ("left"), then split into 80% training and 20% testing sets.
4.Train & Predict — Build a Decision Tree Classifier using entropy criterion, fit it on training data, and predict outcomes on test data.
5.Evaluate & Visualize — Calculate accuracy score, predict on a sample input, and plot the decision tree for visual interpretation.

## Program:
```

Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: 212225040479
RegisterNumber:  VAISHNAVI T

```
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
data = pd.read_csv("Employee.csv")
data = pd.get_dummies(data, drop_first=True)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
plt.figure(figsize=(20,10))

plot_tree(
    model,
    feature_names=X.columns,
    filled=True
)

plt.show()

```
## Output:
<img width="1261" height="641" alt="WhatsApp Image 2026-05-13 at 9 15 12 AM" src="https://github.com/user-attachments/assets/9070b8a2-d91a-4823-9b76-de17731e265b" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
