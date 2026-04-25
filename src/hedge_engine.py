import numpy as np


class HedgeEngine:
    """Engine for computing hedge vectors and adjusting them based on market conditions."""

    def compute_covariance_matrix(self, R_t, sigmas):
        """Compute covariance matrix from correlation matrix and volatilities."""
        D = np.diag(sigmas)
        return D @ R_t @ D

    def compute_multivariate_hedge(self, H_t, ridge_lambda=1e-4, max_weight=1.25):
        """Compute robust optimal hedge vector from covariance matrix."""
        cov_target = H_t[0, 1:]
        cov_hedges = H_t[1:, 1:]

        if cov_hedges.size == 0:
            return np.array([])

        regularized = cov_hedges + np.eye(cov_hedges.shape[0]) * ridge_lambda
        hedge_vector = np.linalg.pinv(regularized) @ cov_target
        return np.clip(hedge_vector, -max_weight, max_weight)

    def confidence_scale(self, H_t, min_confidence=0.05):
        """Scale hedge ratio by average absolute correlation with target asset."""
        if H_t.shape[0] <= 1:
            return 1.0

        var_target = max(H_t[0, 0], 1e-12)
        corr_vals = []
        for i in range(1, H_t.shape[0]):
            var_i = max(H_t[i, i], 1e-12)
            corr = H_t[0, i] / np.sqrt(var_target * var_i)
            if np.isfinite(corr):
                corr_vals.append(abs(corr))

        if len(corr_vals) == 0:
            return min_confidence

        avg_corr = float(np.mean(corr_vals))
        return float(np.clip(avg_corr, min_confidence, 1.0))

    def adjust_for_sentiment(self, hedge_vector, sentiment, strength=0.2):
        """Apply smooth sentiment multiplier using tanh to avoid abrupt jumps."""
        sentiment = float(np.clip(sentiment, -1.0, 1.0))
        multiplier = 1.0 - (strength * np.tanh(sentiment))
        adjusted = hedge_vector * multiplier
        return adjusted, float(multiplier)
