import numpy as np


class HedgeEngine:
    """Engine for computing hedge vectors and adjusting them based on market conditions."""
    
    def compute_covariance_matrix(self, R_t, sigmas):
        """Compute covariance matrix from correlation matrix and volatilities."""
        D = np.diag(sigmas)
        return D @ R_t @ D
    
    def compute_multivariate_hedge(self, H_t):
        """Compute optimal hedge vector from covariance matrix."""
        cov_nifty = H_t[0, 1:]
        cov_hedges = H_t[1:, 1:]
        hedge_vector = np.linalg.inv(cov_hedges) @ cov_nifty
        return hedge_vector
    
    def adjust_for_sentiment(self, hedge_vector, sentiment):
        """Adjust hedge vector based on market sentiment."""
        if sentiment < -0.2:
            return hedge_vector * 1.5
        elif sentiment > 0.2:
            return hedge_vector * 0.8
        return hedge_vector