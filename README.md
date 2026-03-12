# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1 . Import the required libraries.
2 . Upload and read the dataset.
3 . Check for any null values using the isnull() function.
4 . From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5 . Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Venkatesh A
RegisterNumber: 212225040485 
*/
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
data=pd.read_csv("Employee_EX6.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
plt.figure(figsize=(18,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()

```

## Output:
![decision tree classifier model](sam.png)
<img width="1063" height="588" alt="image" src="https://github.com/user-attachments/assets/bb7e328b-d2df-4b6d-9fd9-03114d5e896b" />
<img width="1040" height="557" alt="image" src="https://github.com/user-attachments/assets/53c9326e-64ab-4cfe-8370-eee889e81a87" />
<img width="1057" height="438" alt="image" src="https://github.com/user-attachments/assets/8e84d67a-d7d8-49d5-bd71-ccbd7bd4b93f" />
<img width="801" height="242" alt="image" src="https://github.com/user-attachments/assets/8ee39d8b-a8fa-470e-9137-b93d310dd465" />
<img width="1056" height="503" alt="image" src="https://github.com/user-attachments/assets/ae28b17d-e00c-4d67-aa40-83fb5aeddc2e" />




## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
