import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pickle

df = pd.read_csv(r"C:\Users\sahas\OneDrive\Documents\visual studio code\Data Science projects\Graduate Admission Predictor\Admission_Predict_Ver1.1.csv")

st.set_option('deprecation.showfileUploaderEncoding',False) 
pipe = pickle.load(open('model1.pkl','rb'))

st.sidebar.title("Graduate Admission Predictor")
st.sidebar.image('study.jpg')
user_menu = st.sidebar.radio(
    'Select an Option',
    ('General Section','Overall Analysis')
)

if user_menu == 'General Section':
    st.markdown("<h1 style='text-align: center; color:#ED2B33FF;background-color:black'>Graduate Admission Predictor</h1>", unsafe_allow_html=True)

    #CGPA
    cgpa = st.slider("Input Your CGPA", min_value=0.0, max_value=10.0,  step=0.01)
    #GRE Score
    gre = st.slider("Input Your GRE Score", min_value=0, max_value=340,  step=1)
    #TOEFL Score
    toefl = st.slider("Input Your TOEFL Score", min_value=0, max_value=120,  step=1)
    #university_rating
    rating = st.slider("Rating of the University you wish to get in on a Scale 1-5", min_value=1, max_value=5,  step=1)
    #Letter of Recommendation
    lor = st.slider("Input Your LOR Score", min_value=0.0, max_value=5.0,  step=0.5)
    #Research experience
    experience = st.slider("Have any Prior Research Experience (0 = NO, 1 = YES)", min_value=0, max_value=1,  step=1)


    input_df = [[gre,toefl,rating,lor,cgpa,experience]]

    if st.button('Predict Probability'):
        result =pipe.predict(input_df)
        updated_res = result.flatten().astype(float)
        st.success('The Probability of getting admission is {}'.format(updated_res))

if user_menu == 'Overall Analysis':
    
    st.title('Distribution of University rating (Countplot)')
    fig=plt.figure(figsize=(10,6))
    sns.countplot(x='University Rating',data=df)
    plt.xlabel('University Rating')
    st.pyplot(fig)

    st.title('Distribution of University rating (Pie Plot)')
    fig=plt.figure(figsize=(10,6))
    st.set_option('deprecation.showPyplotGlobalUse', False)
    df['University Rating'].value_counts().plot.pie(autopct='%1.1f%%',figsize=(10,7)) 
    st.pyplot()

    st.title('Distribution of People who have written Research Papers')
    fig = plt.figure(figsize=(10,7))
    sns.countplot(x='Research',data=df,palette='Set1')
    plt.xlabel('Research Papers')
    st.pyplot(fig)