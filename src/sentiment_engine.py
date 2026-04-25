import pandas as pd
import numpy as np
import requests
import os
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
    # 2️⃣ NEWS FETCHING (Using SERP API)
    # ==========================================
    
    def fetch_news(self, days=7):
        
        # Get SERP API key from environment variable
        serp_api_key = "9d2fc6ef0137cd14d84b1f2a8113f2f84ae4acd1de1df8a48d02d1cb49424695"
        if not serp_api_key:
            raise ValueError("SERP_API_KEY environment variable is not set. Please set it with your API key from https://serpapi.com")
        
        queries = [
            "NIFTY 50 India",
            "RBI policy India",
            "Indian stock market",
            "USD INR India",
            "Gold price India",
            "Crude oil India"
        ]
        
        all_results = []
        
        for query in queries:
            try:
                # Use SERP API for news search
                params = {
                    "q": query,
                    "type": "news",
                    "api_key": serp_api_key,
                    "google_domain": "google.co.in",
                    "tbm": "nws",  # News search
                }
                
                response = requests.get("https://serpapi.com/search", params=params)
                response.raise_for_status()
                data = response.json()
                
                # Extract news results
                if "news_results" in data:
                    all_results.extend(data["news_results"])
            
            except requests.exceptions.RequestException as e:
                print(f"Error fetching news for query '{query}': {e}")
                continue
        
        processed_data = []
        current_date = datetime.now()
        
        for item in all_results:
            
            headline = item.get("title", "")
            date_str = item.get("date", "")
            entry_date = current_date
            
            # Try to parse the date from the response
            try:
                if date_str:
                    # SERP API returns dates in various formats, attempt to parse
                    if "ago" in date_str:
                        # Handle relative dates like "2 hours ago", "1 day ago"
                        parts = date_str.split()
                        if len(parts) >= 2:
                            count = int(parts[0])
                            unit = parts[1].lower()
                            
                            if "hour" in unit:
                                entry_date = current_date - timedelta(hours=count)
                            elif "day" in unit:
                                entry_date = current_date - timedelta(days=count)
                            elif "week" in unit:
                                entry_date = current_date - timedelta(weeks=count)
                            elif "month" in unit:
                                entry_date = current_date - timedelta(days=count*30)
            except:
                pass
            
            if headline:  # Only add if headline is not empty
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

