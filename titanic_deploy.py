# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 17:55:20 2022

@author: User
"""

# packages
import os
import pandas as pd
import pickle
import numpy as np

#%% Static Code
# access to the test dataset saved
TEST_PATH = os.path.join(os.getcwd(),'data','test.csv')
# label encoder saved
SEX_LE_PATH = os.path.join(os.getcwd(),'sex_le.pkl')
EMBARKED_LE_PATH = os.path.join(os.getcwd(),'embarked_le.pkl')
# scaler saved
MMS_PATH = os.path.join(os.getcwd(),'mms_scaler.pkl')
# imputer saved
IMPUTER_PATH = os.path.join(os.getcwd(),'knn_imputer.pkl')
# model saved
MODEL_PATH = os.path.join(os.getcwd(),'titanic_model.pkl')
# save new file
NEW_FILE_SAVE_PATH = os.path.join(os.getcwd(),'data','titanic_predicted.csv')

#%% Load pickle files
# load model
model = pickle.load(open(MODEL_PATH,'rb'))
# load label encoder
sex_le = pickle.load(open(SEX_LE_PATH,'rb'))
embarked_le = pickle.load(open(EMBARKED_LE_PATH,'rb'))
# load scaler
scaler = pickle.load(open(MMS_PATH,'rb'))
# load imputer
imputer = pickle.load(open(IMPUTER_PATH,'rb'))

#%% Functions
# use label encoder
def categorical_label_deploy(col_encoder,col_name):
    le= col_encoder
    df2[col_name]= le.transform(df2[col_name])

#%% new input data
# 1) Load new input data
test = pd.read_csv(TEST_PATH)
df = test.copy()

# 2) Data Inspection/Visualization
df.info()
# categorical data: name,sex,ticket.cabin,embarked
df.describe().T
df.boxplot()
# fare consists of great outliers
df.duplicated().sum()
# no duplication
df.isnull().sum()
# null value for age(86), and cabin(327)

# 3) Data Cleaning
# create new feature:Familysize
familysize = df['SibSp']+df['Parch']
# insert new column ast index 12, with the column name and relevant data
df.insert(11,'Familysize',familysize)
# drop unecessary columns
# drop cabin because the missing data is over 50.0%, whcih is 77.1%
df.drop(columns=['PassengerId','Name','SibSp','Parch','Ticket','Cabin'],axis=1,inplace=True)
df2= df.copy()

# change categorical data to label
categorical_label_deploy(col_encoder= sex_le, col_name='Sex')
categorical_label_deploy(col_encoder= embarked_le, col_name='Embarked')
# apply imputation
# trained with Survived column previously
# add empty target column to the position of dataset
df2.insert(0,'Survived',np.nan)
# change data type to allow data imputation
df2['Survived']= pd.to_numeric(df2['Survived'])
df2 =imputer.transform(df2)
# convert df2 to dataframe then rename columns name
df2 = pd.DataFrame(df2)
df2.columns = ['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Familysize']

# 4) Feature Selection
# None

# 5) Data Preprocessing
# apply data scaling for features only
X_data = scaler.transform(df2.iloc[:,1:7])

#%% Deployment
predicted = model.predict(X_data)
predicted = pd.DataFrame(predicted).astype(int)
predicted.columns = ['Survived']
# insert the column predicted to the passengerid column
passengerid= test['PassengerId']
final = pd.concat([passengerid,predicted],axis=1)
# save outcome as new csv file
final.to_csv(NEW_FILE_SAVE_PATH,index=False)

