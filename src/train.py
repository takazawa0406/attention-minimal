import numpy as np
from model import SelfAttention, AveragePooling
from data import generate_data

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def train():
    # ハイパーパラメータ
    T = 8
    d = 8
    n_samples = 200
    lr = 0.1
    epochs = 50

    # データ生成
    X, y = generate_data(n_samples, T, d)

    # モデル
    attention = SelfAttention(d)
    avg = AveragePooling()

    # 分類用重み
    W = np.random.randn(d, 2) * 0.1

    for epoch in range(epochs):
        loss = 0.0
        correct = 0

        for i in range(n_samples):
            seq = X[i]
            label = y[i]

            # Attention
            Y, A = attention.forward(seq)
            z = np.mean(Y, axis=0)

            logits = z @ W
            probs = softmax(logits)

            loss -= np.log(probs[label] + 1e-8)

            if np.argmax(probs) == label:
                correct += 1

        acc = correct / n_samples
        print(f"Epoch {epoch+1:02d} | Loss {loss:.3f} | Acc {acc:.2f}")

if __name__ == "__main__":
    train()

