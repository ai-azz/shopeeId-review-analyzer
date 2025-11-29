# 🛒 Indonesian App Review Analyzer (Shopee Case Study)

> End-to-End NLP Pipeline & Analytics Dashboard: From Raw Scraped Data to LLM-Ready Dataset
> 

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white) ![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

## 📌 Executive Summary

This project architects a robust data pipeline to scrape, clean, and analyze **20,000+ user reviews** from the Shopee Indonesia application. Beyond analysis, this project features a **deployed Streamlit Dashboard** that allows stakeholders to interact with insights and test the sentiment analysis model in real-time.

**The Goal:** To simulate the data preparation phase for an **In-House Large Language Model (LLM)**, transforming unstructured, noisy colloquial Indonesian text ("bahasa gaul") into a structured dataset, while providing actionable business intelligence via an interactive frontend.

### 🚀 Key Achievements

- **Automated Data Pipeline:** Scraped and processed 20,000+ reviews using `google-play-scraper` and a custom Regex+Sastrawi cleaning pipeline.
- **Noise Reduction:** Achieved **93.8% Vocabulary Reduction** (from 184k to 11k tokens), creating a high-quality corpus for LLM training.
- **Interactive Dashboard:** Built a modular Streamlit app with Plotly visualizations to monitor KPIs (Churn Rate, Rating Distribution) and visualize pain points.
- **Real-time Inference:** Integrated a **Logistic Regression Baseline Model (93% Accuracy)** into the dashboard, allowing users to input text and get instant sentiment predictions.

---

## 🖥️ Dashboard Demo

> [Live Demo](https://shopeeid-review-analyzer.streamlit.app/)
>

**Features:**

- **Filters:** Filter reviews by Date, Rating, and Sentiment.
- **Pain Point Analysis:** Interactive WordClouds distinguishing between "Satisfaction Drivers" and "Churn Drivers".
- **AI Playground:** Type your own review and see how the model classifies it.

---

## 🛠️ Project Structure

The project follows a modular software engineering structure for scalability:

```bash
shopee-review-analyzer/
├── .streamlit/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── sentiment_pipeline.pkl
├── notebooks/
│   ├── 01_data_scraping.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda_visualization.ipynb
│   └── 04_model_baseline.ipynb
├── src/
│   ├── app.py
│   └── data_loader.py
├── .gitignore
├── requirements.txt
└── README.md
```

## 🔍 Key Insights (EDA)

Based on the analysis of the cleaned dataset:

1. **The "Logistics" Bottleneck:**
    - Negative reviews are dominated by keywords like `kurir`, `paket`, and `lama`.
    - **Strategic Rec:** The Data Team should correlate `jasa kirim` complaints with specific vendors to flag underperforming logistics partners.
2. **The "Venting" Hypothesis (Review Length):**
    - Negative reviews are **3x longer** (median 13 words) than positive reviews.
    - **LLM Strategy:** Negative reviews provide complex narrative structures ideal for training the model on "Context Understanding," while positive reviews are mostly transactional.

---

## 🔧 Technical Stack

- **Language:** Python 3.10
- **Data Processing:** Pandas, NumPy
- **Visualization:** Plotly Express (Interactive), Matplotlib, Seaborn, WordCloud
- **NLP Tools:** Sastrawi (Stemming), Regex, Emoji
- **Machine Learning:** Scikit-Learn (Logistic Regression, TF-IDF, Pipeline)
- **Web Framework:** Streamlit
- **Environment:** Jupyter Notebook

---

## 💡 How to Run Locally

1. **Clone this repository:**
    ```bash
    git clone https://github.com/ai-azz/shopeeId-review-analyzer.git
    cd shopeeId-review-analyzer
    ```
    
2. **Install requirements:**
It is recommended to use a virtual environment (Conda/Venv).
    ```bash
    pip install -r requirements.txt`
    ```

3. **Run the Streamlit Dashboard:**
Navigate to the `src` directory and run the app.
    
    ```bash
    cd src
    streamlit run app.py
    ```
    
    The application will open in your browser at `http://localhost:8501.