import numpy as np


class HedgeEngine:

    def __init__(self, lambda_sent=0.5, smooth_alpha=0.2):
        self.lambda_sent = lambda_sent
        self.smooth_alpha = smooth_alpha
        self.prev_hedge = None


    def compute_covariance_matrix(self, R_t, sigmas):
        D = np.diag(sigmas)
        return D @ R_t @ D


    def compute_multivariate_hedge(self, H_t):

        cov_nifty = H_t[0, 1:]
        cov_hedges = H_t[1:, 1:]

        # 🔥 Regularization
        cov_hedges += np.eye(cov_hedges.shape[0]) * 1e-6

        # 🔥 Pseudo inverse (stable)
        hedge_vector = np.linalg.pinv(cov_hedges) @ cov_nifty

        # 🔥 Weight constraints
        hedge_vector = np.clip(hedge_vector, -2, 2)

        return hedge_vector


    def adjust_for_sentiment(self, hedge_vector, sentiment_row):

        # sentiment_row: GOLD_sentiment, USDINR_sentiment, CRUDE_sentiment

        sentiment_array = np.array([
            sentiment_row.get("GOLD_sentiment", 0),
            sentiment_row.get("USDINR_sentiment", 0),
            sentiment_row.get("CRUDE_sentiment", 0)
        ])

        adjustment = 1 + self.lambda_sent * sentiment_array

        return hedge_vector * adjustment


    def smooth_hedge(self, hedge_vector):

        if self.prev_hedge is None:
            self.prev_hedge = hedge_vector
            return hedge_vector

        smoothed = (
            (1 - self.smooth_alpha) * self.prev_hedge
            + self.smooth_alpha * hedge_vector
        )

        self.prev_hedge = smoothed

        return smoothed