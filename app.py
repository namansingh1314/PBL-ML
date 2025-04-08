# Import Important Library.

import joblib
import streamlit as st 
from PIL import Image
import pandas as pd


# Load Model & Scaler & Polynomial Features

model=joblib.load('model.pkl')
sc=joblib.load('sc.pkl')
pf=joblib.load('pf.pkl')

# load dataset

df_final=pd.read_csv('test.csv')
df_main=pd.read_csv('main.csv')

# Load Image

image=Image.open('img.png')

# Streamlit Function For Building Button & app.

def main():
    # Slim banner image
    st.image("use.jpg", use_column_width=True, caption="", output_format="auto")

    # Title - smaller and balanced
    st.markdown("<h2 style='text-align:center; color:#2e7d32;'>🌾 Yield Crop Prediction</h2>", unsafe_allow_html=True)

    # Refined green header
    html_temp = '''
    <div style='background-color:#2e7d32; padding:1.5vw; border-radius:0.8vw; margin-bottom: 1.5vw'>
        <h3 style='color:#ffffff; text-align:center; font-size:1.8vw;'>Yield Crop Prediction Machine Learning Model</h3>
    </div>
    <h4 style='color:#2e7d32; text-align:center; font-size:1.4vw;'>🌱 Please Enter Input</h4>
    '''
    st.markdown(html_temp, unsafe_allow_html=True)

    # Inputs in clean layout
    col1, col2 = st.columns(2)

    with col1:
        country = st.selectbox("🌍 Country", df_main['area'].unique()) 
        average_rainfall = st.number_input('🌧️ Average Rainfall (mm/year)', value=None)

    with col2:
        crop = st.selectbox("🌾 Crop", df_main['item'].unique()) 
        presticides = st.number_input('🧪 Pesticides Use (tonnes)', value=None)

    avg_temp = st.number_input('🌡️ Average Temperature (°C)', value=None)

    input = [country, crop, average_rainfall, presticides, avg_temp]
    result = ''

    # Predict Button
    if st.button('🚀 Predict'):
        result = prediction(input)

    # Result Output Box
    if result:
        temp = f'''
        <div style='background-color:#66bb6a; padding:1.5vw; border-radius:0.8vw; margin-top:2vw'>
            <h4 style='color:#003300; text-align:center; font-size:1.6vw;'>Prediction Result: {result}</h4>
        </div>
        '''
        st.markdown(temp, unsafe_allow_html=True)

    # Custom CSS for fine-tuning layout
    st.markdown("""
        <style>
            .stApp {
                background-color: #ffffff;
                color: #2e7d32;
                font-family: 'Segoe UI', sans-serif;
            }
            .stSelectbox label, .stNumberInput label {
                font-weight: 600;
                color: #2e7d32;
            }
            h1, h2, h3, h4 {
                margin-top: 0;
                margin-bottom: 0.6rem;
            }
        </style>
    """, unsafe_allow_html=True)

    
    
    

# Prediction Function to predict from model.
# Albania	Soybeans	1990	7000	1485.0	121.00	16.37
# input=['Albania','Soybeans',1485.0,121.00,16.37]
def update_columns(df, true_columns):
    df[true_columns] = True
    other_columns = df.columns.difference(true_columns)
    df[other_columns] = False
    return df
def prediction(input):
    categorical_col=input[:2]
    input_df=pd.DataFrame({'average_rainfall':input[2],'presticides_tonnes':input[3],'avg_temp':input[4]},index=[0])
    input_df1=df_final.head(1)
    input_df1=input_df1.iloc[:,3:]
    true_columns = [f'Country_{categorical_col[0]}',f'Item_{categorical_col[1]}']
    input_df2= update_columns(input_df1, true_columns)
    final_df=pd.concat([input_df,input_df2],axis=1)
    final_df=final_df.values
    test_input=sc.transform(final_df)
    test_input1=pf.transform(test_input)
    predict=model.predict(test_input1)
    result=(int(((predict[0]/100)*2.47105) * 100) / 100)
    return (f"The Production of Crop Yields:- {result} quintel/acers yield Production. "
            f"That means 1 acers of land produce {result} quintel of yield crop. It's all depend on different Parameter like average rainfall, average temperature, soil and many more.")


if __name__=='__main__':
    main()


