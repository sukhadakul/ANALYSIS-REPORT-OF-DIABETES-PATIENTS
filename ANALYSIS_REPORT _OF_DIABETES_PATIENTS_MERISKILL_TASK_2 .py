#!/usr/bin/env python
# coding: utf-8

# # PROJECT-2: ANALYSIS REPORT OF DIABETES PATIENTS
# Objective:
# The main objective of this project is to diagnostically predict whether a
# patient has diabetes or not.
# Note: The dataset used for this analysis is sourced from the National Institute of
# Diabetes and Digestive and Kidney Diseases.
# About the dataset:
# This dataset is originally from the National Institute of Diabetes and Digestive
# and Kidney Diseases.
# 
# 1. Pregnancies: This variable represents the number of times the patient has
# been pregnant.
# 2. Glucose: It is the plasma glucose concentration measured in milligrams per
# deciliter (mg/dL) of blood. This is a key indicator for diabetes diagnosis.
# 3. Blood Pressure: This variable represents the diastolic blood pressure (mm
# Hg) of the patient.
# 4. Skin Thickness: It indicates the skin thickness (mm) at the triceps area. Skin
# thickness can sometimes be relevant in diabetes diagnosis.
# 5. Insulin: This variable represents the serum insulin level (mu U/ml). Insulin is
# a hormone that regulates blood sugar levels, and its measurement can be
# important in diabetes diagnosis.
# 6. BMI (Body Mass Index): BMI is calculated from the weight and height of
# the patient. It's a measure of body fat and is often used to assess whether a
# person is underweight, normal weight, overweight, or obese.
# 7. Diabetes Pedigree Function: This function is used to represent the diabetes
# pedigree function, which provides information about diabetes mellitus history in
# relatives and genetic influence.
# 8. Age: This variable represents the age of the patient in years.
# 9. Outcome: This is the target variable, and it indicates whether the patient has
# diabetes or not. It is binary, with values 0 and 1, where:
#  - 0 typically indicates that the patient does not have diabetes.
#  - 1 typically indicates that the patient has diabetes.

# # 1) Import necessary libraries 🐍 

# In[30]:


import pandas as pd 
import numpy as np 
import seaborn as sn 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 


# #  2) Load The Dataset 📁
# 

# In[5]:


Data = pd.read_csv("Diabetes_Dataset.csv")
Data 


# In our dataset with 768 rows and 9 columns.

# In[6]:


#Check top 5 records in Our Dataset 
Data.head(5) 


# Top 5 records in Dataset 

# In[7]:


#Check Bottom 5 records in Our Dataset 
Data.tail() 


# Bottom 5 records in our Dataset 

# In[8]:


Data.columns


# In[9]:


# Check sample data in our Dataset
Data.sample() 


# # 3) EDA (Exploratory Data Analysis )🚀

# In[10]:


#Check shape of Dataset 
Data.shape 


# In[11]:


#Check info of Dataset 
Data.info() 


# In[12]:


#Check null value in our Dataset 
Data.isnull().sum() 


# No null value in our Dataset 

# In[13]:


#Check Duplicated values in our dataset 
Data.duplicated().sum()  


# No Duplicate value in our Dataset 

# In[14]:


## Summary statistics 
Data.describe()  


# In[19]:


# Class distribution
Data['Outcome'].value_counts() 


# # 4)Data Visualization 📊📈📉 

# # Pair Plot 📊📈

# In[17]:


# Pairplot to visualize relationships between variables
sn.pairplot(Data, hue='Outcome') 
plt.show()  


# # Heatmap 🌡️

# In[18]:


# Correlation matrix heatmap
corr_matrix = Data.corr()
plt.figure(figsize=(10, 6))
sn.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.show() 


# # 5) Models Training / Building  📈 

# In[21]:


# Split the data into X (features) and y (target)
X = Data.drop('Outcome', axis=1)
y = Data['Outcome'] 


# In[22]:


X 


# In[24]:


y 


# In[25]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 


# In[26]:


# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) 


# In[28]:


# Create a Random Forest Classifier model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train) 


# # 6) Model Evaluation🧪

# In[32]:


# Make predictions
y_pred = model.predict(X_test)
y_pred 


# In[33]:


# Evaluate the model
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred)) 


# # 7) Accuracy 🎯💯🚀

# # I got a Accuracy Score: 0.72 (72%) 

# # Feature Importance 

# In[38]:


feature_importance = clf.feature_importances_
feature_names = X.columns
plt.barh(feature_names, feature_importance)
plt.xlabel("Feature Importance")
plt.title("Feature Importance for Diabetes Prediction")
plt.show() 


# # 8) Conclusion:- 

# • It has a decent level of precision, indicating that when it predicts positive 
# cases (diabetes), it's correct about 65% of the time.

# • The recall value indicates that the model is reasonably effective at 
# identifying actual positive cases, capturing about 62% of them.

# • The F1-score of 0.78suggests that the model provides a balanced 
# performance in terms of precision and recall. 

# • Out of a total of 768 patients, 268 have been diagnosed with diabetes. 

# • A higher number of pregnancies is associated with a decreased 
# likelihood of diabetes. 

# • Patients with above-average blood pressure tend to have a lower 
# likelihood of diabetes. 

# • An increase in blood pressure, BMI, and skin thickness is correlated with 
# an increased likelihood of developing diabetes. 

# • Rising levels of glucose and insulin are linked to an increased risk of 
# diabetes. 

# In[ ]:




