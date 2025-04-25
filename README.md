# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Necessary Libraries and Load Data

2. Split Dataset into Training and Testing Sets

3. Train the Model Using Stochastic Gradient Descent (SGD)

4. Make Predictions and Evaluate Accuracy
  
5. Generate Confusion Matrix


## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Muthulakshmi D
RegisterNumber:  212223040122
*/
```
```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier  
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

iris=load_iris()

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

df.info()
```
![image](https://github.com/user-attachments/assets/02655d3c-8704-4b34-a809-7dc479b2a163)

```
df.head()
```

![image](https://github.com/user-attachments/assets/5a36ffbe-7cc7-4d67-aace-95af21530d99)

```
x=df.drop('target',axis=1)
y=df['target']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
print(x_train.shape)
print(x_test.shape)
```
![image](https://github.com/user-attachments/assets/2f7bc472-db92-424c-95e9-bfd66ee0765a)

```
sc=SGDClassifier()
sc.fit(x_train,y_train)
y_pred=sc.predict(x_test)

print(accuracy_score(y_pred,y_test))
print(confusion_matrix(y_pred,y_test))
```
![image](https://github.com/user-attachments/assets/6e714680-0953-487d-926a-af39239329a2)



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
