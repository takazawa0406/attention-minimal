# Minimal Self-Attention Model (Lecture Assignment)

## 概要

本リポジトリは、 **Self-Attention アルゴリズム**を理解することを目的として、
NumPy を用いて最小構成で実装・実験したものである。

近年は LLM によりコード生成が容易であるため、本課題では **コードそのものよりも、
アルゴリズムの数式的説明と理解**を重視し、モデル構造・計算手順・実装との対応関係を
README 上で明示することに重点を置いた。

---

## 問題設定

長さ \(T\)、特徴次元 \(d\) の系列入力

$$
X = (x_1, x_2, \dots, x_T), \quad x_t \in \mathbb{R}^d
$$

を入力とし、系列全体を 2 クラスに分類する問題を考える。

本実装では、系列の前半と後半の統計量の大小関係に基づいてラベルを付与した
人工データセットを自作して用いた。

---
## モデル構造の概要

本課題で実装したモデルの全体構造を以下に示す。
Self-Attentを最小構成で用い、系列全体を分類する。

```
Input Sequence (T × d)
        │
        ▼
Linear Projection
(Query / Key / Value)
        │
        ▼
Scaled Dot-Product
Self-Attention
        │
        ▼
Output Sequence (T × d)
        │
        ▼
Average Pooling
        │
        ▼
Linear Classifier
        │
        ▼
Prediction
```

## Self-Attention のアルゴリズム

### 1. Query / Key / Value の計算

各時刻 \(t\) の入力ベクトル \(x_t\) に対し、線形変換によって
Query, Key, Value を計算する。

$$
\begin{aligned}
q_t &= x_t W_Q \\
k_t &= x_t W_K \\
v_t &= x_t W_V
\end{aligned}
$$

ここで、

$$
W_Q, W_K, W_V \in \mathbb{R}^{d \times d}
$$

は学習可能な重み行列である。

---

### 2. Scaled Dot-Product Attention

時刻 \(t\) における、時刻 \(s\) への Attention スコアは次式で定義される。

$$
e_{ts} = \frac{q_t \cdot k_s}{\sqrt{d}}
$$

内積を次元数 \(d\) の平方根でスケーリングするのは、
次元が大きくなった際に値が過度に大きくなるのを防ぐためである。

---

### 3. Attention 重みの正規化

Attention スコア \(e_{ts}\) に softmax を適用し、
各時刻 \(t\) における Attention 重みを得る。

$$
\alpha_{ts} =
\frac{\exp(e_{ts})}
{\sum_{j=1}^{T} \exp(e_{tj})}
$$

これにより、各時刻は系列全体の情報を重み付きで参照できる。

---

### 4. 出力系列の計算

Attention 重みを用いて、各時刻の出力ベクトル \(y_t\) を計算する。

$$
y_t = \sum_{s=1}^{T} \alpha_{ts} v_s
$$

この操作により、各時刻の表現は系列全体の情報を統合したものとなる。

---

## 系列全体の集約（Average Pooling）

Transformer では [CLS] トークンを用いることが多いが、
本実装では最小構成と理解の容易さを優先し、平均プーリングを用いた。

$$
z = \frac{1}{T} \sum_{t=1}^{T} y_t
$$

これにより、系列全体を表す固定長ベクトル

$$
\(z \in \mathbb{R}^d\) を得る。
$$

---

## 分類

最終的な分類は線形分類器によって行う。


$$
\hat{y} = z W_C
$$

ここで、

$$
W_C \in \mathbb{R}^{d \times C}
$$

であり、本課題ではクラス数 \(C = 2\) の二値分類を扱った。

---

## 学習と実験結果

50 epoch の学習を行った結果、損失および精度は以下のように推移した。

- Accuracy: 約 0.48  
- Loss: ほぼ一定

---

## 考察

本実装では NumPy を用いた最小構成を目的としており、

- 自動微分を使用していない
- 勾配計算およびパラメータ更新を簡略化している

そのため、Attention の重み行列は十分に更新されず、
分類精度は学習を通して大きく改善しなかった。

しかし、Self-Attention における

- Query / Key / Value の役割
- 内積による系列内関連度の計算
- softmax による正規化
- 系列全体の情報統合

といったアルゴリズムの本質的な構造は、
数式と実装の対応を通じて確認できたと考える。

---

## 実装との対応関係

- Query / Key / Value の計算  
  → `src/model.py : SelfAttention.forward`
- Attention スコアおよび softmax  
  → 同上
- Average Pooling  
  → `src/model.py : AveragePooling`
- 学習ループ  
  → `src/train.py`
- データ生成  
  → `src/data.py`

---

## LLM（ChatGPT）の利用について

本課題において、ChatGPT を以下の目的で利用した。

- Self-Attention に関する代表的な論文（Vaswani et al.）の探索
- README における説明構成や記述内容の整理

アルゴリズムの理解、数式の導出、ならびにコードの実装自体は、
講義内容および自身の理解に基づいて行った。



## 参考文献

- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017  

 

