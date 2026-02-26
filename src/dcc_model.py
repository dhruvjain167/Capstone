import numpy as np

def compute_dcc(std_resids, a=0.02, b=0.95):
    T, N = std_resids.shape
    Q_bar = np.cov(std_resids.T)
    Q_t = Q_bar.copy()
    
    R_t_series = []
    H_t_series = []
    
    for t in range(T):
        z = std_resids[t].reshape(-1, 1)
        Q_t = (1 - a - b)*Q_bar + a*(z @ z.T) + b*Q_t
        
        D_inv = np.diag(1/np.sqrt(np.diag(Q_t)))
        R_t = D_inv @ Q_t @ D_inv
        
        R_t_series.append(R_t)
    
    return np.array(R_t_series)