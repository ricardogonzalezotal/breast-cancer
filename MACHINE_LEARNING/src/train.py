import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

#Carga de datos
df = pd.read_csv('../data/processed/cancer_limpio.csv')

X = df.drop('diagnosis',axis=1)
y = df['diagnosis']

#Separacion train-test
x_train, x_test, y_train, y_test = train_test_split(X,
                                                   y,
                                                   test_size = 0.25,
                                                   random_state=42)


#Escalado de datos
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

#Cargamos modelo
with open('../model/modelo_decisionTree.pkl', 'rb') as archivo:
    mi_modelo = pickle.load(archivo)

mi_modelo.fit(x_train,y_train)
