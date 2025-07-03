# importing the libraries and modules
import pickle
import pandas as pd
#import tensorflow as tf 
#from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from tensorflow.keras.models import load_model
import streamlit as sl 



# Load the model
churn_pred_model = load_model('bank_churn_classification.h5')


# Load encoders and data scalers
with open('label_encoder_gender.pkl', 'rb') as file:
    le_obj_gender = pickle.load(file)

with open('ohe_encoder_geography.pkl', 'rb') as file:
    ohe_obj_geo = pickle.load(file)

with open('standard_scaler.pkl', 'rb') as file:
    ss_obj = pickle.load(file)




# Setting up front-end
sl.title('Bank Customer Churn Prediction')
# Input
geo = sl.selectbox('Geography', ohe_obj_geo.categories_[0])
gender = sl.selectbox('Gender', le_obj_gender.classes_)
age = sl.slider('Age', 18,  90)
balance = sl.number_input('Balance')
cred_score = sl.number_input('Credit Score')
est_salary = sl.number_input('Estimated Salary')
tenure = sl.slider('Tenure', 0, 10)
num_products = sl.slider('Num Of Products', 1, 4)
has_cred = sl.selectbox('Has Cr Card', [0,1])
is_active_mem = sl.selectbox('Is Active Member', [0,1])


# Preparing input data
ip_data = pd.DataFrame(
    {
    'CreditScore': [cred_score],
    #'Geography': [geo],
    'Gender': [le_obj_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_products],
    'HasCrCard': [has_cred],
    'IsActiveMember': [is_active_mem],
    'EstimatedSalary': [est_salary]
    }
)

# Handling OHE columns
geo_encode = ohe_obj_geo.transform([[geo]]).toarray()
geo_df = pd.DataFrame(geo_encode, columns=ohe_obj_geo.get_feature_names_out(['Geography']))
input_data = pd.concat([ip_data.reset_index(drop=True), geo_df], axis=1)


# Scaling the data
pre_processed_input_data = ss_obj.transform(input_data)



# Predicting the output
prediction = churn_pred_model.predict(pre_processed_input_data)
pred_prob = prediction[0][0]



# Printing the results 
sl.write(f'Chrun Probability: {pred_prob:.2f}')
if pred_prob > 0.5:
    sl.write("Customer is likely to churn")
else:
    sl.write("Customer is not likely to churn")
