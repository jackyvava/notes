<center>Lecture 3 正则化和优化</center>

假设我们有N个样本，总共有$j$个类别，每个样本的正确类比为$y_i$，那么我们的总损失函数可以表示为：
$$
L = \frac{1}{N} \Sigma_i^N \Sigma_{j\neq y_i}(f(x_i,w)_j,f(x_i,w)_{y_i})
$$
损失函数不唯一。

![fig1](D:\zjPhD\notes\notes\AI\cs231n\图片\12.png)

那么我们如何去选择是用$w$还是$2w$？

我们选择正则化，正则化可以防止数据过拟合，考虑进噪音。
$$
L(W) = \frac{1}{N} \Sigma_i^N L_i(f(x_i,w)_j,y_i)+\lambda R(W)
$$
$\lambda$是正则化强度，一个超参数。使用正则化可以有效避免过拟合

简单的正则化案例：

$L_1$正则化，$R(W)= \Sigma_k \Sigma_l |W_{k,l}|$

$L_2$正则化，$R(W)= \Sigma_k \Sigma_l W_{k,l}^2$

弹性网络：结合$L_1$和$L_2$，$R(W) =\Sigma_k \Sigma_l \beta W_{l,l}^2+|W_{k,l}| $



为什么要进行正则化？

1. 表达对权重的偏好，比如L2倾向于使得权重值变小，使得模型更加平滑和简单，减少过拟合；而L1倾向于稀疏的权重，即使许多权重变为0，从而进行**特征选择**。
2. 使模型简单，从而在测试数据上表现良好
3. 通过增加曲率改善优化。正则化项会增加损失函数的曲率，使得损失函数更加凸，从而改善优化过程。

案例说明

输入向量 $$x = [1, 1, 1, 1] $$，两个权重向量： 

$$ w_1 = [1, 0, 0, 0] $$ 

$$w_2 = [0.25, 0.25, 0.25, 0.25] $$

无论是$ w_1 $ 还是$ w_2 $，当它们与输入向量$ x $ 相乘时，结果都是 1：

$$ w_1^T x = w_2^T x = 1 $$

其中，$L_2$正则化更加倾向于”分散权重“，使得权重较小且分布均匀。
$$
R(\mathbf{w}_1) = 1^2 + 0^2 + 0^2 + 0^2 = 1
$$

$$
R(\mathbf{w}_2) = 0.25^2 + 0.25^2 + 0.25^2 + 0.25^2 = 0.0625 + 0.0625 + 0.0625 + 0.0625 = 0.25
$$


$L_1$ 正则化更倾向于稀疏的权重，使得许多权重变为0.
$$
R(\mathbf{w}_1) = |1| + |0| + |0| + |0| = 1
$$

$$
R(\mathbf{w}_2) = |0.25| + |0.25| + |0.25| + |0.25| = 1
$$
所以，L_1更加倾向于选择$w_1$，而$L_2$更加倾向于选择$w_2$

- **L2正则化**：在上图中，L2正则化会偏好($\mathbf{w}_2 = [0.25, 0.25, 0.25, 0.25]$，因为它“分散”了权重，减少了正则化项的值。

- **L1正则化**：L1正则化会偏好$ \mathbf{w}_1 = [1, 0, 0, 0]$ ，因为它是稀疏的，符合L1正则化的稀疏性偏好。 

![image-20240715215354896](D:\zjPhD\notes\notes\AI\cs231n\图片\13.png)

现在的问题是我们如何找到最好的W？

引出优化$(Optimization)$

第一种是随机搜索，这种似乎正确率也还可以

```python
bestloss 
for num in range(10000):
    w = np.random(10,3072)
    loss = L(x_train,y_train,W)
    if loss < bestloss:
        bestloss = loss
        bestW = w
    
```

第二种就是沿着梯度斜坡

梯度$\mathbf \nabla f(x)$是函数增长最快的方向

所以一般选取负梯度

![image-20240716011941126](D:\zjPhD\notes\notes\AI\cs231n\图片\14.png)

这种方法是使用数值梯度，非常慢，需要循环所有的维度，而且是近似的

实际上损失函数是权重$W$的函数，可以用微积分来计算$\nabla_W L$

这里$dW = some\ function\ data\ and\ W$

比如我们可以选取前面提到的$L_1$$L_2$正则化这类。

* {数值梯度}}：近似的、慢的、容易编写
* {解析梯度}}：精确的、快的、容易出错

在实践中：总是使用解析梯度，但通过数值梯度来检查实现。这被称为梯度检查



随机梯度下降$(SGD)$
$$
L(W) =\frac1N\sum_{i=1}^NL_i(x_i,y_i,W)+\lambda R(W) \\
\nabla_{W}L(W) =\frac1N\sum_{i=1}^N\nabla_WL_i(x_i,y_i,W)+\lambda\nabla_WR(W) 
$$
当N很大的时候，进行全求和很浪费资源

因此我们使用小批量（minibatch）样本来近似求和，常见的小批量大小为32、64、128

```python
while True:
    data_batch = sample_training_data(data, 256)  # sample 256 examples
    weights_grad = evaluate_gradient(loss_fun, data_batch, weights)
    weights += - step_size * weights_grad  # perform parameter update
```

SGD 存在的问题：

如果损失函数在一个方向上变化很快，再另一个方向上很慢，会出现什么情况

* 在这种情况下，梯度下降在浅维度上进展非常慢，而在陡峭方向上会抖动。

损失函数具有高条件数：Hessian矩阵的最大奇异值与最小奇异值比率很大。

![image-20240716014640740](D:\zjPhD\notes\notes\AI\cs231n\图片\15.png)

SGD 问题2，容易陷入局部最小值和鞍点

SGD问题3，通常使用小批量样本来近似梯度，可能受限于样本选择的影响，包含较大的噪声，这样的噪声下的路径是不稳定的



对于上述解决办法，采用SGD+动量的方法

![image-20240716024655857](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20240716024655857.png)

1. **加速收敛**：动量法可以加速收敛速度，尤其是在鞍点或平坦区域，使得优化过程更快地到达最优解。
2. **避免局部最优**：动量法可以帮助算法摆脱局部最优，继续向全局最优解搜索。
3. **减少抖动**：通过累积梯度信息，动量法可以减少参数更新过程中的抖动，使得优化路径更加平滑稳定。



SGD加动量有两种写法，要适应，见P66



还有更加复杂的优化方法RMSProp:引入历史梯度平方和的移动平均来调整每个维度的梯度更新步长



我们最长用的是Adam优化

给出了Adam优化算法的参数建议：

- `beta1 = 0.9`
- `beta2 = 0.999`
- `learning_rate = 1e-3` 或 `5e-4`

这些值通常是一个很好的起点，可以适用于许多模型。



进一步的，可以使用带有权重衰减的Adam。

下面是标准Adam算法：

```python
first_moment = 0
second_moment = 0
for t in range(1, num_iterations):
    dx = compute_gradient(x)  # 计算梯度
    first_moment = beta1 * first_moment + (1 - beta1) * dx
    second_moment = beta2 * second_moment + (1 - beta2) * dx * dx
    first_unbias = first_moment / (1 - beta1 ** t)
    second_unbias = second_moment / (1 - beta2 ** t)
    x -= learning_rate * first_unbias / (np.sqrt(second_unbias) + 1e-7)

```

标准Adam算法中的$L_2$正则化

```python
dx = compute_gradient(x) + lambda * x  # L2正则化项 lambda * x

```

AdamW算法

```python
x -= learning_rate * (first_unbias / (np.sqrt(second_unbias) + 1e-7) + weight_decay * x)

```

- **AdamW算法的权重衰减**：权重衰减项在参数更新时添加。这种方式使得正则化项与动量和二阶动量估计值无关，从而更加准确地应用权重衰减。

上述的几种算法都将学习率作为了超参数。

对于学习率的设置，有一直恒定不变的，也有衰减学习率的情况

对于衰减学习率：两种常见的策略：步骤衰减（Step Decay）和持续性衰减（Cosine Decay） ^90b6c4

一种是在固定位置进行衰减

一种是采用公式进行衰减：

余弦公式：
$$
\alpha_t = \frac{1}{\alpha_0}(1+cos(\frac{t\pi}{T}))
$$
$T$是总的epoch数

线性公式：
$$
\alpha_t=\alpha_0(1-t/T)
$$
反平方根衰减：
$$
\alpha_t = \alpha_0/(\sqrt t)
$$


### 线性预热（Linear Warmup）P94

线性预热是一种在训练初期逐步增加学习率的方法，防止过高的初始学习率导致损失函数爆炸。具体地，这种方法在训练的前大约5000次迭代中，线性地从0增加学习率。

### 为什么使用线性预热？

- 高初始学习率可能会使损失函数变得不稳定，甚至爆炸。通过线性增加学习率，可以避免这种情况发生，使模型能够平稳地进入训练过程。
- 线性预热有助于模型在初期更稳定地学习，从而提高训练效果。

### 经验法则

图片中提到的经验法则是：

- 如果将批量大小（Batch Size）增加 *N*倍，那么初始学习率也应当增加 *N* 倍。这种做法可以使训练过程在不同的批量大小下保持稳定。 ^28dd99

![image-20240716142955115](D:\zjPhD\notes\notes\AI\cs231n\图片\18.png)



![image-20240716142901588](D:\zjPhD\notes\notes\AI\cs231n\图片\17.png)

两张图分别对比了不同优化方法的工作原理和区别

### 一阶优化（First-Order Optimization）

一阶优化算法只使用目标函数的梯度信息来进行优化。最常见的一阶优化算法是梯度下降（Gradient Descent）。

#### 步骤

1. **使用梯度形成线性近似**：计算当前点的梯度，并用它来线性近似目标函数。
2. **步进以最小化近似值**：沿着梯度的反方向移动一个步长，以减少目标函数值。

#### 图示

- 图中展示了损失函数（Loss）关于权重（$w_1$）的曲线。
- 橙色线表示在某一点的梯度线性近似。
- 红色点表示优化过程中权重的位置，通过沿梯度方向的反方向移动来减少损失。

### 二阶优化（Second-Order Optimization）

二阶优化算法不仅使用梯度信息，还使用<font color='orange'>Hessian矩阵</font>（即==目标函数的二阶导数==）来进行优化。常见的二阶优化算法包括牛顿法（Newton's Method）。

#### 步骤

1. **使用梯度和Hessian矩阵形成二次近似**：计算当前点的梯度和Hessian矩阵，并用它们来二次近似目标函数。
2. **步进到近似值的极小值**：通过二次近似的极小值来更新权重。

#### 图示

- 图中展示了损失函数（Loss）关于权重（$w_1$）的曲线。
- 蓝色曲线表示在某一点的二次近似。
- 红色点表示优化过程中权重的位置，通过沿二次近似的极小值方向移动来减少损失。

使用二阶泰勒展开：
$$
J(\theta) \approx J(\theta_0) + (\theta - \theta_0)^\top \nabla_\theta J(\theta_0) + \frac{1}{2} (\theta - \theta_0)^\top H (\theta - \theta_0)
$$
$\theta_0$ 是当前参数；$\nabla_\theta J(\theta_0)$ 是在 $\theta_0$ 处的梯度 ；$H$ 是在 $\theta_0$ 处的 Hessian 矩阵（目标函数的二阶导数矩阵）

通过求解临界点，我们可以得到牛顿参数更新公式：:
$$
\theta^\star = \theta_0=H^{-1}\mathbf\nabla_{\theta}J(\theta_0)
$$
如何得求解临界点？具体计算如下：



对$\theta$求导
$$
\nabla_\theta J(\theta) \approx \nabla_\theta J(\theta_0) + \nabla_\theta \left( (\theta - \theta_0)^\top \nabla_\theta J(\theta_0) \right) + \nabla_\theta \left( \frac{1}{2} (\theta - \theta_0)^\top H (\theta - \theta_0) \right)
$$

* 常数项 $J(\theta_0)$ 的导数为零

* $(\theta - \theta_0)^\top \nabla_\theta J(\theta_0)$ 的导数：$$\nabla_\theta \left( (\theta - \theta_0)^\top \nabla_\theta J(\theta_0) \right) = \nabla_\theta J(\theta_0).$$

* $\frac{1}{2} (\theta - \theta_0)^\top H (\theta - \theta_0)$ 的导数：$\nabla_\theta \left( \frac{1}{2} (\theta - \theta_0)^\top H (\theta - \theta_0) \right) = H (\theta - \theta_0)$

  这一部分设计<font color='orange'>矩阵求导</font>的结果，对于对称的Hessian矩阵.

求解$\theta $的更新公式
$$
H(\theta-\theta_0) = -\mathbf \nabla_{\theta}J(\theta_0)\\
\theta-\theta_0 = -H^{-1}\mathbf \nabla_\theta J(\theta_0)\\
\theta = \theta_0 - H^{-1}\mathbf \nabla_\theta J(\theta_0)
$$
<font color='red'>为什么这对深度学习不利？</font>

* Hessian有 $O(N^2)$ 个元素
* 计算逆矩阵需要 $O(N^3)$ 的时间复杂度
* $N = \text{（数千万到数亿）}$



### 1. Adam(W) 是一个很好的默认选择

- Adam(W)

  ：Adam优化算法（包括带权重衰减的AdamW）在许多情况下是一个很好的默认选择。

  - **优势**：Adam能够自适应调整每个参数的学习率，并且在处理稀疏梯度和噪声较大的数据时表现良好。
  - **固定学习率**：即使使用固定的学习率，Adam通常也能取得不错的效果，因为它内部的自适应学习率机制能够帮助模型更稳定地收敛。

### 2. SGD + 动量

- SGD + 动量（Momentum）

  ：带有动量的随机梯度下降（SGD with Momentum）在某些情况下可以超过Adam的表现。

  - **优势**：动量可以加速SGD在凹面区域的收敛，并且在一定程度上克服了SGD的振荡问题。
  - **调参需求**：使用SGD + 动量时，通常需要更多的超参数调优，例如学习率和学习率调度策略，以达到最佳性能。

### 3. 如果可以进行全批量更新

- 全批量更新

  ：如果有能力进行全批量更新，那么可以考虑超越一阶优化方法，使用二阶优化方法或更高级的方法。

  - **二阶优化（Second-Order Optimization）**：二阶优化方法利用目标函数的二阶导数信息（Hessian矩阵）来更新参数，理论上收敛速度更快，但计算成本和内存需求较高。
  - **适用性**：由于二阶优化方法的计算复杂度，通常只在能够进行全批量更新（batch size非常大）时考虑使用。

我们上面的优化的是比较简单的情况，那么如果我们想要优化更加复杂的函数呢？



当前，线性损失函数$f=Wx$

下次，我们研究2层的神经网络$f=W_2max(0,W_1x)$

where $x\in \mathbb{R}^D,W_1\in\mathbb{R}^{H\times D},W_2 \in \mathbb{R}^{C \times H}$



二阶优化中的牛顿法，计算量较大，我们可以采用拟牛顿法：

BFGS 方法：

​	BFGS（Broyden–Fletcher–Goldfarb–Shanno）方法是最常用的拟牛顿方法之一。

- **不直接求逆**：BFGS方法不直接计算Hessian矩阵的逆（这需要 𝑂(𝑁3)*O*(*N*3) 的计算量），而是通过逐步更新一个近似的逆Hessian矩阵。
- **秩1更新**：BFGS通过逐步进行秩1更新来近似逆Hessian矩阵，每次更新的计算量是 𝑂(𝑁2)*O*(*N*2)，这比直接求逆要高效得多。



L-BFGS方法：有限内存BFGS方法：

​	L-BFGS（Limited-memory BFGS）方法是BFGS方法的变种，专门用于高维问题，解决了存储和计算的问题。

- **有限内存**：L-BFGS不形成或存储完整的逆Hessian矩阵，而是只存储和更新少量的历史信息（如梯度和参数的变化）。
- **适用性**：由于其有限内存的特点，L-BFGS非常适用于高维度、大规模优化问题，比如深度学习中的优化问题。
- 通常在全批量（Full Batch）、确定性模式下表现非常好，如果有一个单一的确定性的目标函数$f(x)$，那么效果会非常好

> ### 关键点总结
>
> - **拟牛顿法（BFGS）**：通过逐步进行秩1更新来近似逆Hessian矩阵，计算复杂度为 $O(N^2)$，而不是直接求逆的 $O(N^3)$。
> - **有限内存BFGS（L-BFGS）**：不形成或存储完整的逆Hessian矩阵，只存储有限的历史信息，非常适合高维优化问题。

$$
\mathcal{F}(x; \mathbf{w}) \in \mathbb{R}^n
$$

