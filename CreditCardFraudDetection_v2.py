#Importing Libraries

import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 


from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC 
from xgboost import XGBClassifier 

from sklearn.linear_model import LogisticRegression 

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import f1_score 



# Importing CSV Data

df = pd.read_csv('/Users/devrajredkar/Documents/MSC Data Science/Extras/Machine Learning in Python/creditcard.csv')
df.drop('Time', axis = 1, inplace = True)

print(df.head())

total_cases = len(df)
no_fraud = len(df[df.Class == 0])
fraud = len(df[df.Class == 1])
fraud_percent = round(fraud/no_fraud*100, 2)


no_fraud_number_of_cases = df[df.Class == 0]
fraud_number_of_cases = df[df.Class == 1]


sc = StandardScaler()
amount = df['Amount'].values

df['Amount'] = sc.fit_transform(amount.reshape(-1, 1))


# Splitting Data into 80:20 Training and Testing

X = df.drop('Class', axis = 1).values
y = df['Class'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Models

#DecisionTree

tree_model = DecisionTreeClassifier(max_depth = 4, criterion = 'entropy')
tree_model.fit(X_train, y_train)
tree_yhat = tree_model.predict(X_test)

#RandomForestTree

rf = RandomForestClassifier(max_depth = 4)
rf.fit(X_train, y_train)
rf_yhat = rf.predict(X_test)


#KNN

n = 5

knn = KNeighborsClassifier(n_neighbors = n)
knn.fit(X_train, y_train)
knn_yhat = knn.predict(X_test)

#SVM 

svm = SVC()
svm.fit(X_train, y_train)
svm_yhat = svm.predict(X_test)


#XGBoost

xgb = XGBClassifier(max_depth = 4)
xgb.fit(X_train, y_train)
xgb_yhat = xgb.predict(X_test)


#LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_yhat = lr.predict(X_test)



#AccuracyScore

#DecisionTree

accuracy_score(y_test, tree_yhat)

#RandomForestTree

accuracy_score(y_test, rf_yhat)

#KNN

accuracy_score(y_test, knn_yhat)

#SVM 

accuracy_score(y_test, svm_yhat)

#XGBoost

accuracy_score(y_test, xgb_yhat)

#LogisticRegression

accuracy_score(y_test, lr_yhat)



#F1Score

#DecisionTree

f1_score(y_test, tree_yhat)

#RandomForestTree

f1_score(y_test, rf_yhat)

#KNN

f1_score(y_test, knn_yhat)

#SVM 

f1_score(y_test, svm_yhat)

#XGBoost

f1_score(y_test, xgb_yhat)

#LogisticRegression

f1_score(y_test, lr_yhat)



#ConfusionMatrix

#DecisionTree

tree_matrix = confusion_matrix(y_test, tree_yhat, labels = [0, 1])
    
#RandomForestTree

rf_matrix   = confusion_matrix(y_test, rf_yhat, labels = [0, 1])

#KNN

knn_matrix  = confusion_matrix(y_test, knn_yhat, labels = [0, 1]) 

#SVM 

svm_matrix  = confusion_matrix(y_test, svm_yhat, labels = [0, 1]) 

#XGBoost

xgb_matrix  = confusion_matrix(y_test, xgb_yhat, labels = [0, 1])

#LogisticRegression

lr_matrix   = confusion_matrix(y_test, lr_yhat, labels = [0, 1])