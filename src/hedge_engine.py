import numpy as np


class HedgeEngine:
    """Engine for computing hedge vectors and adjusting them based on market conditions."""
    
    def compute_covariance_matrix(self, R_t, sigmas):
        """Compute covariance matrix from correlation matrix and volatilities."""
        D = np.diag(sigmas)
        return D @ R_t @ D
    
    def compute_multivariate_hedge(self, H_t, ridge_lambda=1e-4, max_weight=1.25):
        """Compute robust optimal hedge vector from covariance matrix."""
        cov_nifty = H_t[0, 1:]
        cov_hedges = H_t[1:, 1:]

        if cov_hedges.size == 0:
            return np.array([])

        regularized = cov_hedges + np.eye(cov_hedges.shape[0]) * ridge_lambda
        hedge_vector = np.linalg.pinv(regularized) @ cov_nifty
        return np.clip(hedge_vector, -max_weight, max_weight)
    
    def adjust_for_sentiment(self, hedge_vector, sentiment):
        """Adjust hedge vector based on market sentiment."""
        if sentiment < -0.2:
            return hedge_vector * 1.2
        elif sentiment > 0.2:
            return hedge_vector * 0.9
        return hedge_vector
