# src/data.py
import numpy as np

def generate_data(n_samples=200, T=8, d=8):
    """
    簡単な系列分類データを生成する
    前半と後半の平均の大小でラベルを決める
    """
    X = []
    y = []

    for _ in range(n_samples):
        seq = np.random.randn(T, d)

        if np.mean(seq[:T//2]) > np.mean(seq[T//2:]):
            label = 0
        else:
            label = 1

        X.append(seq)
        y.append(label)

    return np.array(X), np.array(y)

