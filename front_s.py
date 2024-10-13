# import the library
import pickle
import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler


#laad the pickel file 
model=pickle.load(open(r"C:\Users\arati\VS Code_NIt\MLR_house_price_prediction\house price.pkl","rb"))

scaler = StandardScaler()

# tiltle for the app
st.title("House Price Prediction App")

# necessary statement
st.write("To predict the house price, please provide details such as the size, Number of floors, bedrooms, house condition and relevant market data.")

sqft=st.number_input("Enter Squre feet here.",max_value=10000, min_value=400,value=400,step=100)
floor=st.slider("choose floor",0,10)
bedroom=st.slider("Bed room",0,30)
condition = st.slider("House Condition",0,5)


if st.button("Predict Price"):
    sqft_input=np.array([[sqft]])
    floor_input=np.array([[floor]])
    bedroom_input=np.array([[bedroom]])
    condition_input = np.array([[condition]])
    #chage type
     # Convert inputs to floats
    sqft_input = float(sqft)
    floor_input = float(floor)
    bedroom_input = float(bedroom)
    condition_input = float(condition)
    
    # take all input to singe array
    input_data = np.array([[sqft_input, floor_input,bedroom_input,condition_input]])
    user_input_scaled = scaler.fit_transform(input_data)

    prediction=model.predict(user_input_scaled)
    
    #success messege
    st.success(f"The predicted price of the house with {sqft} squre feet, {floor} floors, {bedroom} bed rooms is : ${prediction}")
    
    # last  messege
    st.write("Thanks for chosing our Apps.")
