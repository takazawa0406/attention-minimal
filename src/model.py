import numpy as np

class SelfAttention:
    def __init__(self, d):
        self.d = d
        self.Wq = np.random.randn(d, d) * 0.1
        self.Wk = np.random.randn(d, d) * 0.1
        self.Wv = np.random.randn(d, d) * 0.1

    def forward(self, X):
        """
        X: (T, d)
        """
        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv

        scores = Q @ K.T / np.sqrt(self.d)
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        A = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        Y = A @ V
        return Y, A


class AveragePooling:
    def forward(self, X):
        """
        X: (T, d)
        """
        return np.mean(X, axis=0)

