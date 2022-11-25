import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
import nltk
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer, PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer



st.cache()
def load_data():
    df=pd.read_csv(r'C:\Users\hp\OneDrive\Desktop\project\dataset\1429_1.csv')
    return df





st.set_page_config(
    page_title="NLP for Amazon Customer Reviews",
    layout='centered',
    page_icon="🛃"
)



def get_sentiment(c1,c2,c3):
    if c1 > c2 and c1 > c3:
        return 'positive'
    elif c2> c1 and c2 > c3:
        return 'negative'
    else:
        return 'neutral'




        