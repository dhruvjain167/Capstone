import pandas as pd
import numpy as np
from GoogleNews import GoogleNews
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datetime import datetime, timedelta
import torch


# ==========================================
# 1️⃣ LOAD FINBERT (AUTO GPU DETECTION)
# ==========================================

class FinBERTSentiment:

    def __init__(self):
        print("Loading FinBERT model...")
        
        self.device = 0 if torch.cuda.is_available() else -1
        
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        
        self.pipe = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )
        
        print("FinBERT Loaded Successfully.")


    # ==========================================
    # 2️⃣ NEWS FETCHING
    # ==========================================
    
    def fetch_news(self, days=7):
        
        queries = [
            "NIFTY 50 India",
            "RBI policy India",
            "Indian stock market crash",
            "USD INR India",
            "Gold price India",
            "Crude oil India"
        ]
        
        googlenews = GoogleNews(lang="en", region="IN")
        googlenews.set_period(f"{days}d")
        
        all_results = []
        
        for query in queries:
            googlenews.search(query)
            for page in range(1, 4):
                googlenews.get_page(page)
            
            results = googlenews.results()
            all_results.extend(results)
            googlenews.clear()
        
        processed_data = []
        current_date = datetime.now()
        
        for item in all_results:
            
            headline = item.get("title", "")
            date_str = item.get("date", "")
            entry_date = current_date
            
            try:
                if "hour" in date_str or "min" in date_str:
                    entry_date = current_date
                elif "day" in date_str:
                    days_ago = int(date_str.split(" ")[0])
                    entry_date = current_date - timedelta(days=days_ago)
            except:
                pass
            
            processed_data.append({
                "date": entry_date.date(),
                "headline": headline
            })
        
        df = pd.DataFrame(processed_data).drop_duplicates()
        return df


    # ==========================================
    # 3️⃣ BATCH SENTIMENT SCORING (FAST)
    # ==========================================

    def compute_sentiment(self, news_df, batch_size=16):
        
        if news_df.empty:
            return pd.DataFrame()
        
        print(f"Analyzing {len(news_df)} headlines...")

        headlines = news_df["headline"].astype(str).tolist()
        
        sentiment_scores = []
        
        for i in range(0, len(headlines), batch_size):
            batch = headlines[i:i+batch_size]
            outputs = self.pipe(batch, truncation=True)
            
            for output in outputs:
                label = output["label"].lower()
                score = output["score"]
                
                if label == "positive":
                    sentiment_scores.append(score)
                elif label == "negative":
                    sentiment_scores.append(-score)
                else:
                    sentiment_scores.append(0.0)
        
        news_df["sentiment_score"] = sentiment_scores
        
        daily_sentiment = (
            news_df
            .groupby("date")["sentiment_score"]
            .mean()
            .reset_index()
            .sort_values("date")
        )
        
        return daily_sentiment


    # ==========================================
    # 4️⃣ FULL PIPELINE
    # ==========================================

    def get_daily_sentiment(self, days=7):
        
        news_df = self.fetch_news(days=days)
        
        if news_df.empty:
            print("No news found.")
            return pd.DataFrame()
        
        daily_sentiment = self.compute_sentiment(news_df)
        
        return daily_sentiment

