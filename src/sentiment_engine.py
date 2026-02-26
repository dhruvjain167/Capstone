import pandas as pd
import numpy as np
from gnews import GNews
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datetime import datetime, timedelta
import torch
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)


class FinBERTSentiment:

    def __init__(self):

        self.device = 0 if torch.cuda.is_available() else -1

        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

        self.pipe = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )

        self.gnews = GNews(language='en', country='IN', max_results=50)
        self.asset_queries = {
                "NIFTY": "Indian stock market OR NIFTY OR Sensex",
                "GOLD": "Gold price India OR MCX Gold",
                "USDINR": "USD INR OR Indian rupee OR forex India",
                "CRUDE": "Crude oil India OR Brent oil OR WTI oil"
        }


    def _safe_date(self, published, fallback):
        try:
            dt = pd.to_datetime(published, errors="coerce")
            if pd.isna(dt):
                return fallback
            return dt.to_pydatetime()
        except:
            return fallback


    def fetch_asset_news(self, asset, days=7):

        end = datetime.now()
        start = end - timedelta(days=days)

        query = self.asset_queries[asset]

        articles = self.gnews.get_news(query)
        records = []

        for article in articles:

            title = article.get("title", "")
            if not title:
                continue

            published = article.get("published date")
            dt = self._safe_date(published, end)

            if start <= dt <= end:
                records.append({
                    "date": dt.date(),
                    "headline": title
                })

        return pd.DataFrame(records).drop_duplicates()


    def compute_sentiment(self, df, batch_size=16):

        if df.empty:
            return pd.DataFrame()

        texts = df["headline"].astype(str).tolist()
        scores = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            outputs = self.pipe(batch, truncation=True)

            for o in outputs:
                label = o["label"].lower()
                score = o["score"]

                if label == "positive":
                    scores.append(score)
                elif label == "negative":
                    scores.append(-score)
                else:
                    scores.append(0.0)

        df = df.copy()
        df["score"] = scores

        daily = df.groupby("date")["score"].mean().to_frame()

        return daily


    def get_asset_sentiment(self, days=7):

        all_sentiment = {}

        for asset in self.asset_queries.keys():

            df = self.fetch_asset_news(asset, days)
            daily = self.compute_sentiment(df)

            daily.columns = [f"{asset}_sentiment"]

            all_sentiment[asset] = daily

        # Merge all asset sentiment series
        final_df = None

        for df in all_sentiment.values():
            if final_df is None:
                final_df = df
            else:
                final_df = final_df.join(df, how="outer")

        return final_df.fillna(0)