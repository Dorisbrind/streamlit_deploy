import streamlit as st
import pickle
import pandas as pd

df=pd.read_csv('Video_Games.csv')
st.header('Video games predictor')
platforms=df['Platform'].unique()
genres=df['Genre'].unique()
developers=df['Developer'].unique()
ratings=df['Rating'].unique()
with st.sidebar:
    #1 Platform
    platform=st.selectbox('1. Entrer la platforme',platforms)
    #2 Genres
    genre=st.selectbox('2. Entrer le genre',genres)
    #3 Developer
    developer=st.selectbox('3. Entrer le developeur',developers)
    #4 Ratings
    rating=st.selectbox('4. Entrer le rating',ratings)
    #5  User score
    critic_score=float(st.slider('Entrer le critic_score',13.0,98.0))
    #6 Critc score
    user_score=float(st.slider('Entrer le critic_score',0.0,9.7))
    #Critic_Score	Critic_Count	User_Score

st.subheader('Recap du dataframe')

input_df=pd.DataFrame([[platform,genre,developer,rating,critic_score,user_score]],\
                      columns=['Platform','Genre','Developer','Rating','Critic_Score','User_Score'])
st.write(input_df)
model=pickle.load(open('XGBRegressor-model','rb'))

button=st.button('Predict')
if button:
    st.write(model.predict(input_df))