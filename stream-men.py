import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('best_men_model.sav')

# Load data
data = pd.read_csv('MEN_SHOES.csv')

# Application title
st.title('Men Shoes Rating Prediction App')

# Sidebar for user inputs
st.sidebar.header('User Input Parameters')

def user_input_features():
    brand_name = st.sidebar.selectbox('Brand Name', data['Brand_Name'].unique())
    product_details = st.sidebar.selectbox('Product Details', data['Product_details'].unique())
    how_many_sold = st.sidebar.number_input('How Many Sold', value=0)
    current_price = st.sidebar.number_input('Current Price', value=0.0)
    
    data_dict = {
        'Brand_Name': brand_name,
        'Product_details': product_details,
        'How_Many_Sold': how_many_sold,
        'Current_Price': current_price
    }
    features = pd.DataFrame(data_dict, index=[0])
    return features

df = user_input_features()

# Display user inputs
st.subheader('User Input parameters')
st.write(df)

# Make prediction
if st.button('Predict'):
    try:
        prediction = model.predict(df)
        # Convert prediction to float
        prediction = float(prediction[0])
        st.subheader('Prediction')
        st.write(f'Estimated Rating: {prediction:.2f}')
    except Exception as e:
        st.error(f'Error making prediction: {e}')
