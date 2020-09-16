#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 02:41:24 2020

@author: emmanuelmoudoute-bell
"""

# Import libraries
import pandas as pd
import numpy as np

# Import data
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]

# Encode categorical data and scale continuous data
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
preprocess = make_column_transformer(
        (OneHotEncoder(), ['Geography', 'Gender']),
        (StandardScaler(), ['CreditScore', 'Age', 'Tenure', 'Balance',
                            'NumOfProducts', 'HasCrCard', 'IsActiveMember', 
                            'EstimatedSalary']))
X = preprocess.fit_transform(X)
X = np.delete(X, [0,3], 1)


# Séparation du jeu de donées en train et test 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)


# Changement d'échelle
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler() 
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Partie 2 - Contruire le réseau de neuronnes

# Importation des modules de keras
import keras 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialisation du réseau de neuronnes
classifier =  Sequential()

# Ajouter la couche d'entrée et une couche cachée
classifier.add(Dense(units= 6, activation="relu",
                     kernel_initializer = "uniform", input_dim=11))
#permet d'éviter le sur-entrainement il faut l'ajouter sur chaque couche crée
classifier.add(Dropout(rate=0.1))

# Ajouter une deuxieme couche cachée
classifier.add(Dense(units= 6, activation="relu",
                     kernel_initializer = "uniform"))
classifier.add(Dropout(rate=0.1))
# Ajout de la couche de sortie
classifier.add(Dense(units= 1, activation="relu",
                     kernel_initializer = "uniform"))

classifier.compile(optimizer="adam", loss="binary_crossentropy", 
                   metrics=["accuracy"])

# Entrainer le réseau de neuronnes
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# Prédiction sur le jeu de test
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)



# Predicting a single new observation
"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""
Xnew = pd.DataFrame(data={
        'CreditScore': [600], 
        'Geography': ['France'], 
        'Gender': ['Male'],
        'Age': [40],
        'Tenure': [3],
        'Balance': [60000],
        'NumOfProducts': [2],
        'HasCrCard': [1],
        'IsActiveMember': [1],
        'EstimatedSalary': [50000]})
Xnew = preprocess.transform(Xnew)
Xnew = np.delete(Xnew, [0,3], 1)
new_prediction = classifier.predict(Xnew)
new_prediction = (new_prediction > 0.5)

## Faire la matrice de confusion
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
  classifier =  Sequential()
  classifier.add(Dense(units= 6, activation="relu",
                       kernel_initializer = "uniform", input_dim=11))
  classifier.add(Dense(units= 6, activation="relu",
                       kernel_initializer = "uniform"))
  classifier.add(Dense(units= 1, activation="relu",
                       kernel_initializer = "uniform"))
  classifier.compile(optimizer="adam", loss="binary_crossentropy", 
                     metrics=["accuracy"])
  return classifier

# k -fold cros validation
classifier = KerasClassifier(build_fn=build_classifier,batch_size=10, epochs=100)
precision = cross_val_score( estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs= -1)

moyenne = precision.mean()
ecart_type = precision.std()

# Partie 4

# Importation des modules
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
  classifier =  Sequential()
  classifier.add(Dense(units= 6, activation="relu",
                       kernel_initializer = "uniform", input_dim=11))
  classifier.add(Dropout(rate=0.1))
  classifier.add(Dense(units= 6, activation="relu",
                       kernel_initializer = "uniform"))
  classifier.add(Dropout(rate=0.1))
  classifier.add(Dense(units= 1, activation="relu",
                       kernel_initializer = "uniform"))
  classifier.compile(optimizer=optimizer, loss="binary_crossentropy", 
                     metrics=["accuracy"])
  return classifier
 
classifier = KerasClassifier(build_fn=build_classifier)
paramaters = {"batch_size": [25, 32],
              "epochs": [100, 500],
              "optimizer": ["adam", "rmsprop"]}
grid_search = GridSearchCV(estimator= classifier, 
                           param_grid=paramaters,
                           scoring="accuracy",
                           n_jobs=-1,
                           cv=10)
grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_precision = grid_search.best_score_
