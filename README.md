# Self-Attention Minimal Implementation

## 概要
本リポジトリでは、講義で扱った Self-Attention 機構を
全結合層のみからなる最小構成で実装し、
簡単な系列分類問題に適用した。

Transformer 全体の実装ではなく、
Attention 機構そのものの理解を目的としている。

## 目的
- Self-Attention の計算内容をコードとして理解する
- 平均プーリングとの違いを比較する
- Attention が系列中のどの位置を重視するかを可視化する

## モデル構成
- 入力系列長: 8
- 各要素の次元数: 8
- Attention head: 1
- 使用層: 全結合層のみ
- 活性化関数: ReLU / Softmax

## Self-Attention の計算
入力系列を X とすると、以下の計算を行う。

Q = X W_Q  
K = X W_K  
V = X W_V  

Attention 重み A は次式で求める。

A = softmax( Q K^T / sqrt(d) )

出力は次のように計算される。

Y = A V

## 実装内容
- NumPy を用いて Self-Attention を自作実装
- 系列分類タスクのための簡単なデータを生成
- 平均プーリングモデルとの性能比較を実施

## 実験結果
- Attention を用いたモデルは、重要な系列位置に重みを集中させることが確認できた
- 平均プーリングよりも高い分類精度を達成した
- Attention 重みの可視化により、モデルの振る舞いを直感的に理解できた

（figures/attention.png に結果を保存）

## 実行方法
```bash
pip install -r requirements.txt
python src/train.py
