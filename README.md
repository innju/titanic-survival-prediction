# titanic-survival-prediction
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)
![Spyder](https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon)
![Heroku](https://img.shields.io/badge/heroku-%23430098.svg?style=for-the-badge&logo=heroku&logoColor=white)
  
This analysis aimed to create a prediction model that predict the survival of titanic's passengers.
<br>The python scripts are tested and run on Spyder(Python 3.8).
<br>Prediction submitted to Kaggle gained public score of 0.78229.
<br>Simple web application is created using Streamlit and Heroku at the end of the analysis.

### DATA SOURCE:
Original sources of the data can be found in the link below:
<br>[Titanic- Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data?select=train.csv)
<br>Thanks to the Kaggle for organizes the titanic competition.

### FOLDERS AND FILES UPLOADED:
**data folder**: training dataset, testing dataset and predicted data
<br>**figures folder**: classification report,pairplots and the interface of the application
<br>**titanic_train.py**: training file
<br>**titanic_deploy.py**: prediction based on test dataset
<br>**embarked_le.pkl**: label encoder saved for the embarked variable
<br>**sex_le.pkl**: label encoder saved for the sex variable
<br>**knn_imputer.pkl**: imputer saved to the file
<br>**mms_scaler.pkl**: scaler saved to the file
<br>**titanic_model.pkl**: best model saved to the file

Files needed for Heroku deployment with Streamlit file
<br>**titanic_streamlit.py**: application created using Streamlit
<br>**Procfile**: commands that need to be executed by the Heroku on startup.
<br>**requirements.txt**: versions of relevant libraries required in titanic_streamlit.py
<br>**setup.sh**: create a Streamlit folder with a config.toml file


### ABOUT THE MODEL:

A new feature named Familysize is created by sum up the 'SibSp' and 'Parch'.
<br>Pairplots are plotted to understand the distribution of data and the relationship between variables.

![Image](https://github.com/innju/titanic-survival-prediction/blob/main/figures/pairplot_titainic.png)

<br>Then, relationship between categorical variables and categorical target variable are further justified using chi square test.
<br>Relationship between continuous variables and categorical target variable are justified using ANOVA test.
<br>Features selected are **'Pclass','Sex','Age','Fare','Embarked', and 'Familysize'**. 
<br>Machine learning pipelines is used to compare between different classification model.
<br>Best model identified for this analysis is the **random forest model**, with **accuracy of 85%**.

![Image](https://github.com/innju/titanic-survival-prediction/blob/main/figures/rf_titanic.png)

### VIEW THE STREAMLIT APP:
If you have anaconda installed on your device, you can view the application by follow the steps below:
<br>Open Anaconda prompt > conda activate (name of ur environment) > cd (main folder path) > streamlit run (file name for deployment)
<br>Outcome of the app is either Not survive or Survive.
<br>The interface of the app is shown as below:

![Image](https://github.com/innju/titanic-survival-prediction/blob/main/figures/streamlit_titanic.png)

Based on the input data, it is able to predict the survival of passenger in the titanic tragic.

### VIEW THE APP DEPLOYED TO HEROKU
Please click at the link below to view the app:
[Titanic survival prediction](https://titanic-survival-heroku.herokuapp.com/)
<br>You may refers to [Deploying a basic Streamlit app to Heroku](https://towardsdatascience.com/deploying-a-basic-streamlit-app-to-heroku-be25a527fcb3) for detailed steps on deployment. Thanks to Mr. Muralidhar.


<br>Thanks for reading.
