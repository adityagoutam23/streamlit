import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from math import sqrt
from sklearn.preprocessing import LabelEncoder
plt.rcParams['figure.figsize'] = (12.0, 9.0)


class A:
    def __init__(self):
        self.count = 0

    def teamA(self):
        st.set_option('deprecation.showPyplotGlobalUse', False)
        
        st.title("Regression Models")
        st.header("Made by: Aditya Goutam")
        uploaded_file =st. file_uploader("Choose a CSV file")
        #using the dataset for linear regression
        #df = pd.read_csv(f'{uploaded_file}')
        
        df = pd.read_csv('/Users/Coding/Documents/CSE/AI_ML/dataset/data.csv')

        st.header("Choose a Regression model to predict the values")

        model_reg = st.selectbox(" ", ["Linear Regression", "Logistic Regression", "Multivalue Regression"], key=self.count)
        self.count += 1
        #val = st.slider("Filter using Years of experience", 0.00, 10.5)
        #df = df.loc[df["YearsExperience"] >= val]
        if model_reg == "Linear Regression":
            df = df[['a', 'b']]
            df.head()
            st.write('Head of the data frame')
            st.write(df.head())
            df.corr()
            st.write('Correlation for the data frame')
            st.write(df.corr())
            df.fillna(method ='ffill', inplace = True)
            X = np.array(df['a']).reshape(-1, 1)
            y = np.array(df['b']).reshape(-1, 1)
              
            # Separating the data into independent and dependent variables
            # Converting each dataframe into a numpy array 
            # since each dataframe contains only one column
            df.dropna(inplace = True)
              
            # Dropping any rows with Nan values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=5)
              
            # Splitting the data into training and testing data
            regr = LinearRegression()
              
            regr.fit(X_train, y_train)
            st.write('Accuracy of the Linear Regression model is')
            st.write(regr.score(X_test, y_test))
            
            st.header("Predict the value!")
            usr_pred = st.number_input('Enter the value for which the prediction is to be made: ')
            result_lr = regr.predict([[usr_pred]])
            if st.button("Predict"):
                st.success(f"Predicted value is {result_lr}")
            
            
            y_pred = regr.predict(X_test)
            plt.scatter(X_test, y_test, color ='b')
            plt.plot(X_test, y_pred, color ='k')
            st.write('Data scatter of predicted values')  
            plt.show()
            st.pyplot()
            # Data scatter of predicted values
            
            st.header('Error Calculation!')
            
            mae = mean_absolute_error(y_true=y_test,y_pred=y_pred)
            mse = mean_squared_error(y_true=y_test,y_pred=y_pred)
            rmse = mean_squared_error(y_true=y_test,y_pred=y_pred,squared=False)
              
            st.write(f'MAE: {mae}')
            st.write(f'MSE: {mse}')
            st.write(f"RMSE: {rmse}")
            
        if model_reg == "Logistic Regression":
            df = pd.read_csv("/Users/Coding/Documents/CSE/AI_ML/dataset/LR.csv")
            df.head()
            st.write('Head of the data frame')
            st.write(df.head())
            df.corr()
            st.write('Correlation for the data frame')
            st.write(df.corr())
            X_train, X_test, y_train, y_test = train_test_split(df.drop(['Final_Decision','Object'],axis=1), df['Final_Decision'], test_size=0.30,random_state=101)
            logmodel = LogisticRegression()
            logmodel.fit(X_train,y_train)
            predictions = logmodel.predict(X_test)
            accuracy=confusion_matrix(y_test,predictions)
            accuracy=confusion_matrix(y_test,predictions)
            accuracy=accuracy_score(y_test,predictions)
            st.write('Accuracy of the Logistic Regression model is')
            st.write(accuracy)
            height = st.number_input('Enter the height: ')
            distance = st.number_input('Enter the distance: ')
            LNL = st.number_input('Enter weather living or non living: ')
            YD = st.number_input('Enter the YOLO decision: ')
            SR = st.number_input('Enter the Speech Recognition decision: ')
            result = logmodel.predict([[height,distance,LNL,YD,SR,1]])
            if st.button("Predict"):
                st.success(f"Predicted value is {result}")            
                
            y_pred = logmodel.predict(X_test)
            
            giniEval = [] #To Store the Different Scores so as to compare at Last
            confusionMatrix = confusion_matrix(y_test, y_pred)
            
            st.header('Score!')
            
            st.write("\nTesting Using Gini Index")
            st.write(f"\n\tThe Confusion Matrix : {confusionMatrix}\n")
            
            st.write(f"\n\tThe F1 Score : {f1_score(y_test, y_pred)}")
            giniEval.append(f1_score(y_test, y_pred))
            
            st.write(f"\n\tThe Accuracy Score is : {accuracy_score(y_test, y_pred)}")
            
            giniEval.append(accuracy_score(y_test, y_pred))
            st.write(f"\n\tThe Precision Score is : {precision_score(y_test, y_pred)}")
            
            giniEval.append(precision_score(y_test, y_pred))

        if model_reg == "Multivalue Regression":
            df = pd.read_csv('/Users/Coding/Documents/CSE/AI_ML/dataset/Metro.csv')
            df.head()
            st.write('Head of the data frame')
            st.write(df.head())
            df.corr()
            st.write('Correlation for the data frame')
            st.write(df.corr())
            df["price_per_m2"] = df["Price"]/df["Area"]
 
            # Creating a instance of label Encoder.
            le = LabelEncoder()
             
            # Using .fit_transform function to fit label
            # encoder and return encoded label
            label = le.fit_transform(df['Location'])
            
            df.drop("Location", axis=1, inplace=True)
             
            # Appending the array to our dataFrame
            # with column name 'Purchased'
            df["Location"] = label
            
            df.head()
            st.write('Head of the new data frame')
            st.write(df.head())
            
            x = df[["Location", "Area"]]
            y = df["price_per_m2"]
            
            # splitting dataset into Train set and Test set
            x_train,x_test,y_train,y_test = train_test_split(x,y, train_size = 0.75 , random_state=0)
            
            model1 = LinearRegression()
            model1.fit(x_train,y_train)
            
            
            user_input_mr1 = st.number_input('Enter the first value for prediction: ')
            user_input_mr2 = st.number_input('Enter the second value for prediction: ')
            result1 = model1.predict([[user_input_mr1,user_input_mr2]])
            if st.button("Predict"):
                st.success(f"Predicted value is {result1}") 
                
            Y_pred = model1.predict(x_test)
            
            st.header('Error Calculation!')
            
            mae_training = mean_absolute_error(y_test, Y_pred)
            st.write(f"Test MAE: {round(mae_training,2)}")
            
            mse_training = mean_squared_error(y_test, Y_pred)
            st.write(f"Test MSE: {round(mse_training,2)}")
            

            st.write(f"Test RMSE: {(sqrt(mse_training))}")
            
            
            

    def main(self):
        self.teamA()


obj = A()
obj.teamA()

