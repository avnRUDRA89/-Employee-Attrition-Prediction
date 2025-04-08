import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import seaborn as sns
import pickle
import os
import streamlit as st
from project_2 import *



uploadedDataset = 0

def main():
    choice = st.sidebar.selectbox("Pages", ("Home", "Data Visualisation", "Predict"))
    
    if choice == "Home":
        st.image('logo.png', width = 200)
        st.header("𝔸𝕂ℝ")
        st.write("Retain your talent, don't let them slip away - predict and prevent employee attrition with our powerful tool.")
        
        uploadedDataset = st.file_uploader("Upload your dataset", type={'csv'})
        
        if uploadedDataset:
            df = pd.read_csv(uploadedDataset)
            st.dataframe(df.head(30))
    
    
    if choice == "Data Visualisation":
        st.title("Employee Attrition Prediction")

        # Load the data
        st.header("Data")
        st.write(attrdata)

        # Preprocess the data
        st.header("Preprocessed Data")
        st.write(dataset)
        
        st.header('Data Cleaning')
        st.dataframe(location_dict_new)
        

        # Bar chart showing the number of employees who left vs stayed
        st.subheader("Employee Attrition")
        attrition_counts = attrdata["Stay/Left"].value_counts()
        fig, ax = plt.subplots()
        sns.barplot(x=attrition_counts.index, y=attrition_counts.values, ax=ax)
        ax.set_title("Number of Employees who Left vs Stayed")
        ax.set_xlabel("Stay/Left")
        ax.set_ylabel("Number of Employees")
        st.pyplot(fig)
        
        st.header("Confusion Matrix: ")
        
        confusion_mat = confusion_matrix(y_test, y_pred)
        labels = ['Class 0', 'Class 1', 'Class 2']
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(confusion_mat, annot=True, cmap='Blues', fmt='g', xticklabels=labels, yticklabels=labels, ax=ax)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        print(f"Accuracy: {accuracy}")
        print(f"Confusion matrix:\n{confusion_mat}")
        st.pyplot(fig)
        
       
        
        st.header("Accuracy: ")
        st.image('accuracy.png', use_column_width=True)
        
        
        
    if choice == "Predict":
        
        st.header("Prediction")
        
        st.subheader("So the percentage of attrition is :-% ")
        
        if st.button("Calculate"):
            st.success(attrition_percentage)
            
        
        

if __name__ == "__main__":
    main()
