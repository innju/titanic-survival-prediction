# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 16:39:56 2022

This python script is used to build a predictive model that answers the 
questions: "what sorts of people were more likely to survive?" 

@author: User
"""

# packages
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.impute import KNNImputer
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import chi2_contingency

#%% static code
# load data path
TRAIN_PATH = os.path.join(os.getcwd(),'data','train.csv') 
# save label encoder 
SEX_LE_PATH = os.path.join(os.getcwd(),'sex_le.pkl')
EMBARKED_LE_PATH = os.path.join(os.getcwd(),'embarked_le.pkl')
# save scaler
MMS_PATH = os.path.join(os.getcwd(),'mms_scaler.pkl')
# save imputer
IMPUTER_PATH = os.path.join(os.getcwd(),'knn_imputer.pkl')
# save model
MODEL_PATH = os.path.join(os.getcwd(),'titanic_model.pkl')

#%% Functions
# save label encoder
def categorical_label(col_name,SAVE_PATH):
    le= LabelEncoder()
    df2[col_name] = le.fit_transform(df2[col_name])
    pickle.dump(le,open(SAVE_PATH,'wb'))

#%% EDA
# 1) Load data
train = pd.read_csv(TRAIN_PATH)


# 2) Data inspection/visualization
df= train.copy()
df.head(10)
df.info()
# 5 categorical data= ['Name','Sex','Ticket','Cabin','Embarked']
# consist of some missing data
df.columns
# by logic, PassengerId, Name, and Ticket won't be needed in training data
# Some data consists of only number and no label at the front
# Name title could be useful to represent social status
# however, not necessary because pclass already represent the social status
# Sum of Sibsp and Parch could represent new feature: family size
df.describe().T
df.boxplot()
# great outliers on Fare
df.isnull().sum()
# missing data for age(177,19.9%),cabin(687,77.1%) and embarked(2,0.2%)
df.duplicated().sum()
# no duplicated data
df['Survived'].value_counts()
# 0:549,1:342 =>0:62%,1:38%
# about 38% of the passengers survived


# 3) Data cleaning
# create new feature:Familysize
familysize = df['SibSp']+df['Parch']
# insert new column ast index 12, with the column name and relevant data
df.insert(12,'Familysize',familysize)
# drop unecessary columns
# drop cabin because the missing data is over 50.0%, whcih is 77.1%
df.drop(columns=['PassengerId','Name','SibSp','Parch','Ticket','Cabin'],axis=1,inplace=True)
# drop whole rows with NaN for Embarked column since the missing data is just 2
# not logic if impute for embarked column
df2 = df[df.Embarked.notnull()]
# recheck missing data
df2.isnull().sum()
# no more missing data

# change categorical data to label
df2.info()
# Categorical data: Sex, Embarked
# save pickle file
categorical_label(col_name='Sex',SAVE_PATH= SEX_LE_PATH)
categorical_label(col_name='Embarked',SAVE_PATH= EMBARKED_LE_PATH)

# deal with missing data: Age
# dependent variable is included to prevent biased estimates
imputer = KNNImputer()
# by default, n_neighbors=5
# fit and transform for whole dataset
df2= imputer.fit_transform(df2)
# save pickle file
pickle.dump(imputer,open(IMPUTER_PATH,'wb'))

# convert df2 to dataframe then rename columns name
df2 = pd.DataFrame(df2)
df2.columns = df.columns


# 4) Feature selection
# pairplots
plt.figure()
sns.pairplot(data=df2, hue='Survived')
plt.show()
# at first glance, higher survival rate when passengers from upper class,
# female, younger age, higher fare, smaller family size and departed from Cherbourg
# based on historical news, titanic first stop is at Cherbourg
# passengers might have longer time to get familar with the ship structure and
# know where to go if emergency happens
# hence, no features excluded

#%% further check for relationship
# change the datatype from float to int for relevant features
df2['Pclass'] = df2['Pclass'].astype(int)
df2['Sex'] = df2['Sex'].astype(int)
df2['Embarked'] = df2['Embarked'].astype(int)
df2['Familysize'] = df2['Familysize'].astype(int)
df2['Survived'] = df2['Survived'].astype(int)

# relationship between categorical variables and categorical target variable
# chisquaretest
# p-value<0.05 => dependent
# convert data into a contigency table with frequencies
contingency_pclass = pd.crosstab(df['Pclass'],df['Survived'])
c, p, dof, expected = chi2_contingency(contingency_pclass)
print(p) #4.55e-23
contingency_sex = pd.crosstab(df['Sex'],df['Survived'])
c, p, dof, expected = chi2_contingency(contingency_sex)
print(p) #1.20e-58
contingency_embarked = pd.crosstab(df['Embarked'],df['Survived'])
c, p, dof, expected = chi2_contingency(contingency_embarked)
print(p) #1.77e-06
contingency_familysize = pd.crosstab(df['Familysize'],df['Survived'])
c, p, dof, expected = chi2_contingency(contingency_familysize)
print(p) #3.58e-14


# relationship between continuos variables and categorical target variable
# ANOVA
# continuous variable: fare,age
anova_fare = ols('Survived ~ Fare',data=df2).fit()
sm.stats.anova_lm(anova_fare,typ=2) # typ refer to type of anova test to perform
# p-value obtained: 7.03e-71
anova_age =  ols('Survived ~ Age',data=df2).fit()
sm.stats.anova_lm(anova_age,typ=2)
# p=value obtained: 8.90e-29
# since p-value<0.05=> both features have significant influence on Survived


# 5) Data preprocessing
# data scaling for features only
mms = MinMaxScaler()
df2_features = mms.fit_transform(df2.iloc[:,1:7])
# save pickle file
pickle.dump(mms,open(MMS_PATH,'wb'))

# convert df2_features to dataframe then rename columns 
df2_features = pd.DataFrame(df2_features)
df2_features.columns = ['Pclass', 'Sex', 'Age','Fare','Embarked','Familysize']
X= df2_features # features
y = df2['Survived'] #target

print(X.shape) #(889,6)
print(y.shape) #(889,)
# train,test and split data
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

#%% Model
# ML pipeline with different classifier
steps_NB = [('NB',GaussianNB())]
steps_SVM = [('SVM',svm.SVC())]
steps_tree = [('Tree',DecisionTreeClassifier())]
steps_forest = [('Forest',RandomForestClassifier())]
steps_logis = [('Logis',LogisticRegression(solver='liblinear'))]

# create pipeline
pipeline_NB = Pipeline(steps_NB) #To load the steps into the pipeline
pipeline_SVM = Pipeline(steps_SVM) 
pipeline_tree = Pipeline(steps_tree)
pipeline_forest = Pipeline(steps_forest)
pipeline_logis = Pipeline(steps_logis)
#create a list to store all the created pipelines
pipelines= [pipeline_NB, pipeline_SVM,pipeline_tree,pipeline_forest,pipeline_logis]


#fitting of data
for pipe in pipelines:
    pipe.fit(x_train, y_train)
    
pipe_dict = {0:'NB', 1:'SVM', 2:'Tree', 3:'Forest',4:'Logistic'}

#%% Performance evaluation
# find out the best model
# view the classification report and confusion matrix of all model
for i,model in enumerate(pipelines):
    y_pred = model.predict(x_test)
    target_names = ['Died', 'Survived']
    print(pipe_dict[i])
    print(classification_report(y_test, y_pred,target_names=target_names))
    print(confusion_matrix(y_test, y_pred))
# Random forest model achieves highest accuracy, 0.85%

#%% Save the best model
pickle.dump(pipeline_forest,open(MODEL_PATH,'wb'))
