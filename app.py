import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
import nltk

import plotly.express as px
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer, PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def remove_nan(text):
    if isinstance(text,str):
        return text
    else:
        return ""

def sentiment_score(a, b, c):
    if (a>b) and (a>c):
        print("Positive 😊 ")
    elif (b>a) and (b>c):
        print("Negative 😠 ")
    else:
        print("Neutral 🙂 ")

st.cache()
def load_data():
    stop=stopwords.words('english')
    st = PorterStemmer()
    sia=SentimentIntensityAnalyzer()
    df=pd.read_csv(r'C:\Users\hp\OneDrive\Desktop\project\dataset\1429_1.csv')
    columns = ['id','name','keys','manufacturer','reviews.dateAdded', 'reviews.date','reviews.didPurchase',
        'reviews.userProvince', 'reviews.dateSeen', 'reviews.doRecommend','asins',
        'reviews.id', 'reviews.numHelpful', 'reviews.sourceURLs', 'reviews.title','reviews.userCity']
    df = pd.DataFrame(df.drop(columns,axis=1,inplace=False))
    df['reviews.text'] = df['reviews.text'].apply(remove_nan)
    df['clean_review']=df['reviews.text'].apply(lambda x: ' '.join([word for word in x.split() if word not in(stop)]))
    df['clean_review'] = df['clean_review'].str.replace('[^\w\s]','')
    df['clean_review'] = df['clean_review'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
    df["Positive"] = [sia.polarity_scores(i)["pos"] for i in df["clean_review"]]
    df["Negative"] = [sia.polarity_scores(i)["neg"] for i in df["clean_review"]]
    df["Neutral"] = [sia.polarity_scores(i)["neu"] for i in df["clean_review"]]
    return df

st.set_page_config(
    page_title="NLP for Amazon Customer Reviews",
    layout='centered',
    page_icon="🛃"
)
st.title('Sentiment Analysis Of Amazon Customer Reviews')
with st.spinner("Loading data, this may take time..."):
    df = load_data()
options1 = ['View data', 'View Analysis', 'View Visualization']
ch = st.selectbox("select an option", options1)

if ch == options1[0]:

    st.dataframe(df)

if ch == options1[1]:
    st.header("Overall sentiment analysis of the reviews")
    X = sum(df["Positive"])
    Y = sum(df["Negative"])
    Z = sum(df["Neutral"])
    st.write("Positive: ", X)
    st.write("Negative: ", Y)
    st.write("Neutral: ", Z)
    sentiment_score(X, Y, Z)
    Scores=['Positive','Negative','Neutral']
    
    st.header("Showing the sentiment of the review")
    def get_sentiment(c1,c2,c3):
        if c1 > c2 and c1 > c3:
           return 'positive'
        elif c2> c1 and c2 > c3:
           return 'negative'
        else:
           return 'neutral'
    df['sentiment']=df[['Positive','Negative','Neutral']].apply(lambda x: get_sentiment(x.Positive,x.Negative,x.Neutral),axis=1)
    cat_count_df = df.groupby(['categories','sentiment'], as_index=False)['reviews.text'].count()
    st.dataframe(df)

    st.header("Subjectivity of the reviews")
    subjectivity=[]
    for i in df['clean_review'].values:
        try:
            analysis=TextBlob(i)
            subjectivity.append(analysis.sentiment.subjectivity)
        except:
            subjectivity.append(0)   
    df['subjectivity']=subjectivity
    st.dataframe(df)

if ch == options1[2]:
    st.header("Graphical representation of sentiment of reviews")
    X = sum(df["Positive"])
    Y = sum(df["Negative"])
    Z = sum(df["Neutral"])
    Scores=['Positive','Negative','Neutral']

    data=[X,Y,Z]
    fig = px.pie(names=Scores,values=data)
    st.plotly_chart(fig)

    st.header("Word Cloud for reviews")
    st.image('output.png', use_column_width=True)
    st.header("Word Cloud for reviews with rating 5")
    st.image('output1.png', use_column_width=True)
    st.header("Word Cloud for reviews with rating 4")
    st.image('output2.png', use_column_width=True)
    st.header("Word Cloud for reviews with rating 3")
    st.image('output3.png', use_column_width=True)
    st.header("Word Cloud for reviews with rating 2")
    st.image('output4.png', use_column_width=True)
    st.header("Word Cloud for reviews with rating 1")
    st.image('output5.png', use_column_width=True)