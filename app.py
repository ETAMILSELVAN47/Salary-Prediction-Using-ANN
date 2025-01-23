import streamlit as st
import tensorflow
from tensorflow.keras.models import load_model
import pickle
import pandas as pd



# Export models
import os
preprocessor_dir='preprocessor'

label_encoder_obj_file_path=os.path.join(preprocessor_dir,'label_encoder.pkl')
onehotencoder_obj_file_path=os.path.join(preprocessor_dir,'onehot_encoder.pkl')
scaler_obj_file_path=os.path.join(preprocessor_dir,'scaler.pkl')

model_dir='model'
model_file_path=os.path.join(model_dir,'model.h5')

## Load the ANN model
model=load_model(model_file_path)

## Label Encoder
with open (file=label_encoder_obj_file_path,mode='rb') as file:
    label_encoder_obj=pickle.load(file=file)


## OneHotEncoder
with open(file=onehotencoder_obj_file_path,mode='rb') as file:
    onehot_encoder_obj=pickle.load(file=file)


## Standardization
with open (file=scaler_obj_file_path,mode='rb') as file:
    scaler_obj=pickle.load(file=file)   

st.title('ANN Salary Prediction')

#userinput:
credit_score=st.number_input('Credit Score')
gender=st.selectbox('Gender',label_encoder_obj.classes_)
age=st.slider('Age',18,92)
tenure=st.slider('Tenure',0,10)
balance=st.number_input('Balance')
num_of_products=st.slider('Number of Products',1,4)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])
exited=st.selectbox('Exited',[0,1])
geography=st.selectbox('Geography',onehot_encoder_obj.categories_[0])


input_data={'CreditScore': credit_score,
 'Gender': gender,
 'Age': age,
 'Tenure': tenure,
 'Balance': balance,
 'NumOfProducts': num_of_products,
 'HasCrCard': has_cr_card,
 'IsActiveMember': is_active_member,
 'Exited': exited,
 'Geography':geography}

df=pd.DataFrame(data=[input_data])

#Apply LabelEncoder on Gender
df['Gender']=label_encoder_obj.transform(df['Gender'])


#Apply OHE on Geography
geo_encoded=onehot_encoder_obj.transform(df[['Geography']]).toarray()
geo_encoded_df=pd.DataFrame(data=geo_encoded,
             columns=onehot_encoder_obj.get_feature_names_out(input_features=['Geography']))

df=pd.concat(objs=[df,geo_encoded_df],axis=1)

# Drop Geography column
df.drop(columns='Geography',inplace=True)

# Standardization
input_scaled=scaler_obj.transform(df)

# Prediction
output=model.predict(input_scaled)

st.write(f'Estimated Salary is {output[0][0]:.2f}')