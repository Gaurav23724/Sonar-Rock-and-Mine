# create model of detaction of rock and  mine 
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# load dataset
data = pd.read_csv("sonar data.csv")
# separate  Rock and Mine 
# Rock = data[data.R == 'R']
# Mine = data[data.R == 'M']
# undersample  balance the classes
# Mine_sample = Mine.sample(n=len(Rock), random_state=2)
# data = pd.concat([Mine_sample, Rock], axis=0)

# split data 
x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values
# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder

le  = LabelEncoder()
y  = le.fit_transform(y)
# split data into training and testing sets

x_train , x_test , y_train , y_test = train_test_split( x , y , test_size = 0.1 , stratify= y , random_state = 1 )
# train logistic regression mode
classifire = LogisticRegression()
classifire.fit(x_train , y_train)
# evaluate model performance
# Accuracy of model
train_acc = accuracy_score(classifire.predict(x_train), y_train)
test_acc = accuracy_score(classifire.predict(x_test), y_test)


# y_pred = classifire.predict(x_test)
# print(accuracy_score(y_pred , y_test))

# create Streamlit app
st.title("Sonar Rock and Mine Detection Model")
st.write("Enter the following features to find out Rock and Mine :")

# create input fields for user to enter feature values
input_df =(st.text_input('Input All features'))
input_df_lst = input_df.split(',')
# create a button to submit input and get prediction
submit = st.button("Submit")
if submit:
    # get input feature values
    features = np.array(input_df_lst, dtype=np.float64)
    # make prediction
    prediction = classifire.predict(features.reshape(1,-1))
    # display result
    if prediction[0] == 1:
        st.write("Rock Are Exist ")
    else:
        st.write("Mine Are Exist")