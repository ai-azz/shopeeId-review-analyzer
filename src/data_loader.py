import pandas as pd
import streamlit as st
import joblib
import plotly.express as px

# caching data to prevent reloading every time
@st.cache_data
def load_data():
    df = pd.read_csv('../data/processed/shopee_reviews_clean.csv')
    
    if 'review_date' in df.columns:
        df['review_date'] = pd.to_datetime(df['review_date'])
        
    df['text_clean'] = df['text_clean'].fillna('').astype(str)
    
    if 'sentiment' not in df.columns:
        def categorize(rating):
            if rating >= 4: return 'Positive'
            elif rating == 3: return 'Neutral'
            else: return 'Negative'
        df['sentiment'] = df['rating'].apply(categorize)
    
    return df

# load model with cache resource
@st.cache_resource
def load_model():
    model = joblib.load('../models/sentiment_pipeline.pkl')
    return model

# helper for plotly
def plot_sentiment_distribution(df):
    fig = px.pie(df, names='sentiment', title='Sentiment Distribution',
                 color='sentiment',
                 color_discrete_map={'Positive':'#28a745', 'Negative':'#dc3545', 'Neutral':'#ffc107'},
                 hole=0.4)
    return fig

def plot_rating_bar(df):
    rating_counts = df['rating'].value_counts().sort_index()
    fig = px.bar(x=rating_counts.index, y=rating_counts.values, 
                 labels={'x':'Stars', 'y':'Count'},
                 title='Rating Distribution',
                 color=rating_counts.values, color_continuous_scale='Oranges')
    fig.update_layout(xaxis_type='category')
    return fig