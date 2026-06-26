import numpy as np
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.datasets import load_wine

mlflow.set_tracking_uri("http://127.0.0.1:5000")

dataset=load_wine()
X= dataset.data
y=dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

max_depth=2
n_estimators=3

# mlflow.set_experiment('yt-experiment')# you do this or 

with mlflow.start_run(experiment_id=2):
    clf=RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators)
    clf.fit(X_train,y_train)

    y_pred=clf.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)
    
    mlflow.log_metric('accuracy',accuracy)
    mlflow.log_param('max depth',max_depth)
    mlflow.log_param('n_estimators',n_estimators)

    #creating a confusion matrix 
    cm=confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=dataset.target_names,yticklabels=dataset.target_names)
    plt.ylabel('actual')
    plt.xlabel('Predicted')
    plt.title('Confusion matrix')

    plt.savefig('confusion-matrix.png')

    mlflow.log_artifact('confusion-matrix.png')
    mlflow.log_artifact(__file__)
    print(accuracy)