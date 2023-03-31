# Bandit
implementations of algorithms for bandit problem

<br>

# Algorithms

## Probabilistic bandit problem
各アームが平均 $\mu$ について定まる分布 $\pi(\mu)$ に従って報酬が与えられるbandit問題。

<br>

### ・Epsilon-greedy
$\epsilon$の確率でランダムにアームを選択し、 $1-\epsilon$ の確率で報酬の標本平均に対して貪欲にアームを選択する方策。

<br>

### ・UCB
UCBスコア $\bar{\mu}_i(t) := \hat{\mu}_i(t) + \sqrt{\frac{\log t}{2N_i(t)}}$ が最大となるアームを選択する方策。

$O(1/t)$ の範囲で真の期待値の信頼区間を構成し、大きめに期待値を見積もる事でregretの理論限界を達成する。

UCBスコアはヘフディングの不等式を用いて信頼区間の上限を計算する事で導出される。

<br>

### ・Thompson sampling
履歴 $H(t)$ について、 $\tilde{\mu}_i \sim \pi(\mu_i | H(t))$ を生成し、 $\tilde{\mu}_i$ が最大となるアームを選択する方策。

UCB方策における尤度を事後分布に置き換えた物。


<br><br>



## 最適腕識別

### ・逐次削除方策
信頼上限(UCB)と信頼下界(LCB)それぞれのスコアを用いる。

各ステップでの最適アームのLCBスコアをUCBスコアが下回るアームを削除し、
残っているアームを引き続ける方策。


### ・LUCB方策
上と同様にUCB/LCBスコアを用いる。

各ステップでの最適腕 $\hat{i}^\ast_{t}$ と $\hat{i}^\circ_t=argmax_{i \neq \hat{i}^\ast_{t}}\bar{\mu}_i(t)$ の2つを引いていく方策。
