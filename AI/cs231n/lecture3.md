<center>Lecture 3 正则化和优化</center>

假设我们有N个样本，总共有$j$个类别，每个样本的正确类比为$y_i$，那么我们的总损失函数可以表示为：
$$
L = \frac{1}{N} \Sigma_i^N \Sigma_{j\neq y_i}(f(x_i,w)_j,f(x_i,w)_{y_i})
$$
损失函数不唯一。

![fig1](D:\zjPhD\notes\notes\AI\cs231n\图片\12.png)

那么我们如何去选择是用$w$还是$2w$？