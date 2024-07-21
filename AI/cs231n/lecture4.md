<center>Lecture 4 神经网络和反向传播</center>

# 回顾

![image-20240720153624975](D:\zjPhD\notes\notes\AI\cs231n\图片\19.png)

当前有的：

1. 对应的$(x,y)$
2. 分数函数
3. 损失函数

$$
L_i = -log(\frac{e^{sy_i}}{\Sigma_j e^s j})\\
L_i = \Sigma_{j\neq y_i}max(0,s_j-s_{y_i}+1)\\
L = \frac{1}{N}\Sigma_{i=1}^N L_i+R(W)
$$

SoftMax严格来说是对结果先取e再看其在总体所占比例

数值解法：慢，近似，但是容易写

微积分方法：快，精确，但是容易出错

通常是采用微积分方法，然后用数值解法检查



梯度下降

[[lecture3#^28dd99|随机梯度下降问题及解决]]：

全部N丢进去会比较昂贵，所以我们通常丢进去32/64/128的minibatch

[[lecture3#^90b6c4|学习率的更新方式]]：步骤衰减（Step Decay）和持续性衰减（Cosine Decay）

# 深度学习

以前：$f=wx$

现在$f=w_2 max(0,w_1x)$

式中：$x\in \mathbb{R}^D,w_1\in\mathbb{R}^{H\times D},w_2\in\mathbb{R}^{C\times H}$

Neural network 也有其他的名字：$fully-connected\ networks\ or\ multi-layer\ perceptrons$

also 3-layer Neural Network:
$$
f = W_3max(0,W_2max(0,W_1x))\\
x\in\mathbb{R}^D,W_1\in\mathbb{R}^{H_1\times D},W_2\in\mathbb{R}^{H_2\times H_1},W_3\in\mathbb{R}^{C\times H_2}
$$
$max(0,z)$is called activated function which preventing end up with linear classifier again
$$
W_4 = W_3W_2W_1\in \mathbb{R}^{C\times H_2\cdot  H_2\times H_1\cdot H_1\times D}=\mathbb{R}^{C\times D}
$$
我们相当于又得到了线性层

```python
import numpy as np
from numpy.random import randn
N, D_in, H, D_out = 64, 1000, 100, 10
x, y =randn(N, D_in), randn(N, D_out)

w_1, w_2 = randn(D_in, H), randn(H, D_out)
for t in range(2000):
    h = 1/(1+np.exp(-x.dot(w_1)))
    y_pred = h.dot(w_2)
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h.T.dot(grad_y_pred)
    grad_h = grad_y_pred.dot(w_2.T)
    grad_w1 = x.T.dot(grad_h * h * (1 - h))

    w_1 -= 1e-4 * grad_w1
    w_2 -= 1e-4 * grad_w2
```

我们的核心是取找到合适的权重，如果我们能够计算出$\frac{\partial L }{\partial W_1},\frac{\partial L}{\partial W_2}$就能逐渐更新权重。



那么我们如何取算这个梯度呢？

1. 比较坏的方法是用纸演算，这种浪费时间和纸
2. 更好的想法：计算图和反向传播

## 反向传播

