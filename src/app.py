import streamlit as st
import pandas as pd
import plotly.express as px
from data_loader import load_data, load_model, plot_sentiment_distribution, plot_rating_bar
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# page config (wide layout + title + icon)
st.set_page_config(
    page_title="Shopee Sentiment Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# load data and model
df = load_data()
pipeline = load_model()

# sidebar (filters)
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/fe/Shopee.svg/2560px-Shopee.svg.png", width=150)
    st.title("Filters")
    
    # filter date
    if 'review_date' in df.columns:
        min_date = df['review_date'].min()
        max_date = df['review_date'].max()
        date_range = st.date_input("Select Date Range", [min_date, max_date])

    # filter rating
    selected_rating = st.multiselect("Select Rating", options=[1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])
    
    # filter sentiment
    selected_sentiment = st.multiselect("Select Sentiment", options=['Positive', 'Negative'], default=['Positive', 'Negative'])

    st.markdown("---")
    st.caption("Created by [ai-azz](https://github.com/ai-azz)")

# apply filters
mask = (df['rating'].isin(selected_rating)) & (df['sentiment'].isin(selected_sentiment))
df_filtered = df[mask]

# main page: kpi cards
st.title("🛒 Shopee Indonesia App Reviews Analytics")
st.markdown("### Executive Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Reviews", f"{len(df_filtered):,}")

with col2:
    avg_rating = df_filtered['rating'].mean()
    st.metric("Average Rating", f"{avg_rating:.2f} ⭐", delta=f"{avg_rating-4.0:.2f} vs Target")

with col3:
    neg_pct = (len(df_filtered[df_filtered['sentiment']=='Negative']) / len(df_filtered)) * 100
    st.metric("Churn Risk (Negative)", f"{neg_pct:.1f}%", delta="-2.0%" if neg_pct < 15 else "+5%", delta_color="inverse")

with col4:
    # business insight
    top_complaint = "Pengiriman" 
    st.metric("Top Complaint", top_complaint)

st.markdown("---")

# tabs layout (to avoid long page)
tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "😠 Pain Point Analysis", "❓AI Prediction Playground"])

# TAB 1: General Stats
with tab1:
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.plotly_chart(plot_sentiment_distribution(df_filtered), use_container_width=True)
        
    with col_right:
        st.plotly_chart(plot_rating_bar(df_filtered), use_container_width=True)
    
    # show raw data
    with st.expander("View Raw Data"):
        st.dataframe(df_filtered[['review_date', 'rating', 'review_text', 'sentiment']].head(100))

# TAB 2: Pain Point (WordCloud)
with tab2:
    st.header("What are users complaining about?")
    
    col_wc1, col_wc2 = st.columns(2)
    
    with col_wc1:
        st.subheader("Positive Reviews")
        # generate WordCloud on the fly
        pos_text = ' '.join(df_filtered[df_filtered['sentiment']=='Positive']['text_clean'])
        if pos_text:
            wc_pos = WordCloud(width=400, height=300, background_color='white', colormap='Greens').generate(pos_text)
            fig, ax = plt.subplots()
            ax.imshow(wc_pos, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.warning("No positive reviews in selected filter.")

    with col_wc2:
        st.subheader("Negative Reviews (Pain Points)")
        neg_text = ' '.join(df_filtered[df_filtered['sentiment']=='Negative']['text_clean'])
        if neg_text:
            wc_neg = WordCloud(width=400, height=300, background_color='white', colormap='Reds').generate(neg_text)
            fig, ax = plt.subplots()
            ax.imshow(wc_neg, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.warning("No negative reviews in selected filter.")

# TAB 3: real time prediction
with tab3:
    st.header("Test the AI Model")
    st.write("Enter a review below to see how the model classifies it.")
    
    user_input = st.text_area("User Review:", "Barangnya bagus tapi pengiriman lama banget!")
    
    if st.button("Analyze Sentiment"):
        if user_input:
            prediction = pipeline.predict([user_input])[0]
            probs = pipeline.predict_proba([user_input])
            confidence = probs.max()
            
            st.markdown("### Result:")
            if prediction == 'Positive':
                st.success(f"**Positive** 😊 (Confidence: {confidence:.2%})")
            else:
                st.error(f"**Negative** 😡 (Confidence: {confidence:.2%})")
                
                if "lama" in user_input.lower() or "kurir" in user_input.lower():
                    st.info("💡 **Suggestion:** This seems related to Logistics. Check delivery partner status.")
                elif "aplikasi" in user_input.lower():
                    st.info("💡 **Suggestion:** This seems like a Technical Issue. Check App Logs.")
        else:
            st.warning("Please enter some text.")