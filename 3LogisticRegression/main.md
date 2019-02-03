# Logistic Regression
## Introduction
The purpose of classification is to classify. For given training data set $D = \{x_i,y_i\}_{i=1}^N,y_i\in C,C = \{1,...,c\}$, the algorithm's output is to name the specific category of a new given data input.
$$
\hat{y} = f(x)
$$
The output satisfies to be a Bernoulli Variable.
$$
p(y;\mu) = \mu^y(1-\mu)^{(1-y)}\\
p(y=1) = \mu, \quad p(y=0)=1-\mu\\
\mu(x) = \sigma(w^T x)
$$
$\sigma(x)$, sigmoid function is to transform the range of input to $[0,1]$. A commonly used sigmoid function is:
$$
\sigma(z) = \frac{1}{1+e^{-z}}
$$
Hence, in Logistic Models, the logarithm odds of a certain classification is:
$$
\ln\frac{p(y=1|x)}{p(y=0|x)} = \ln(e^{w^T x}) = w^T x\\
When\quad\frac{p(y=1|x)}{p(y=0|x)} > 1,\quad w^T x > 0,\quad y = 1\\
Otherwise,\quad w^T x < 0,\quad y=0\\
When\quad w^T x = 0,\text{ the object lies on the boundary.}
$$
## Loss Function
As in the case of linear regression, here we must decide the goal of the function. Namely, which function has to be minimized.
### 0/1 Loss
$$
L(y,\hat{y}) = \begin{cases}
    0 & y = \hat{y}\\
    1 & y \ne \hat{y}
\end{cases}
$$
Such a loss function can accurately tell if the model had made a right choice. However, the loss function is not continuous, hence not realistic for calculations. We must find surrogate loss functions instead. The surrogate function tends to be convex and some how consistent with the 0/1 loss function.
### Maximum Likelihood Estimation
Write Bernoulli Distribution continuously:
$$
y|x~Bernoulli(\mu (x)),p(y|x;\mu (x)) = \mu (x)^y(1-\mu (x))^{(1-y)},\mu (x) = \sigma (w^Tx)
$$
Likelihood function:
$$
l(\mu) = \ln p(D) = \sum_{i=1}^N \ln p(y_i|x_i) = \sum_{i=1}^N\ln (\mu (x_i)^{y_i}(1-\mu (x_i))^{1-y_i}) = \sum_{i=1}^N
y_i\ln(\mu(x_i)) + (1-y_i)\ln(1-\mu(x_i))
$$
Loss function:
$$
J(\mu) = -l(\mu)
$$
Known as Minus Log Likelihood Loss, also Logistic Loss.
### Cross Entropy Loss
The function of Logistic Loss can also be interpreted with the concept of Cross Entropy. Cross Entropy, namely, tells the difference between the 2 distributions.

- Suppose it is known that $y=1$, $y|x~Bernoulli(1)$.
- Suppose $\hat{y} = 1$ has probability $\mu(x)$, $\hat{y}|x~Bernoulli(\mu(x))$
- By definition, the Cross Entropy is:
$$
CE(y,\mu(x)) = -\sum_y p(y|x)\ln p(\hat{y}|x) = \begin{cases}
    -\ln\mu(x) & if \quad y = 1\\
    -\ln(1-\mu(x)) & otherwise
\end{cases}
$$
## Regular
In a Logistic Model, we always have the loss function of Logistic Loss Function. Target function:
$$
J(w;\lambda) = \sum_{i=1}^N L(y_i,\mu(x_i;w)) + \lambda R(w)
$$
$R(w)$ can be L1, L2, or L1+L2
L1:
$$
||w||_1 = \sum_{j=1}^D |w_j|
$$
L2:
$$
||w||_2^2 = \sum_{j=1}^2 w_j^2
$$
In Scikit-Learn, the constant $\lambda$ is differently placed.
$$
J(w;\lambda) = C\sum_{i=1}^N L(y_i,\mu(x_i;w)) + R(w)
$$
## Newton's Method
Similar to Gradient Descending Method, this is another way of finding the minimum.
### Single Variable Case
- Take an approximation $x^*$ for $f(x^*)=0$;
- Take first degree Taylor Expansion of f;
$$
f(x) = f(x_t) + (x - x_t)f'(x_t)
$$
- Suppose $x_{t+1}$ is a better approximation;
$$
f(x_{t+1}) = f(x_t) + (x_{t+1} - x_t)f'(x_t),\quad x_{t+1} = x_t - \frac{f(x_t)}{f'(x_t)}
$$
Compute the function for a few times, the null point can be quickly found. By searching for the null point of first derivative of the loss function, the minimum can be easily found.
### Multi-variable Case
- Let gradient be the first derivative and Hessian Matrix to be the second derivative;
$$
H(x) = \begin{pmatrix}
    \frac{\partial ^2f}{\partial x_1^2} & \frac{\partial ^2f}{\partial x_1 \partial x_2} & ... & \frac{\partial ^2f}{\partial x_1 \partial x_D} \\
    \frac{\partial ^2f}{\partial x_1 \partial x_2} & \frac{\partial ^2f}{\partial x_2^2} & ... & \frac{\partial ^2f}{\partial x_2 \partial x_D} \\ \vdots & \vdots & \ddots & \vdots \\
    \frac{\partial ^2f}{\partial x_1 \partial x_D} & \frac{\partial ^2f}{\partial x_D \partial x_2} & ... & \frac{\partial ^2f}{\partial x_D^2}
\end{pmatrix}\\
x_{t+1} = x_t - H^{-1}(x_t)\nabla f(x_t)
$$
### In Conclusion
1. $t=0, x_0$ set to be a random number;
2. Compute the gradient and Hessian Matrix of f at $x_t$;
3. Compute moving direction: $d_t = H^{-1}_tg_t\Leftarrow H_td_t=g_t$;
4. Compute the new x: $x_{t+1} = x_t - d_t$;
5. Check if the result is close enough.
## Quasi-Newton Method
- In some cases with so many variables, the Hessian Matrix is either to hard to compute or unable to remain positive definite and not invertible.
- Hence, we have some quasi method, by using not the second derivatives to construct a Hessian Matrix or its inverse matrix.
### Valid Circumstances
- After $t$ times of iteration, we get result $x_{t+1}$;
- Write down the second degree Taylor Expansion:
$$
f(x) = f(x_{t+1}) + (x - x_{t+1})\nabla f(x_{t+1}) + \frac{1}{2}(x - x_{t+1})^T\nabla ^2 f(x_{t+1})(x - x_{t+1})
$$
- Take gradient of both sides:
$$
\nabla f(x) = \nabla f(x_{t+1}) + \nabla^2f(x_{t+1})(x - x_{t+1})
$$
- Take $x = x_t$, $g_t = \nabla f(x_t)$, $H_t = \nabla^2f(x_t)$:
$$
g_{t+1} - g_t = H_{t+1}(x_{t+1} - x_t)
$$
- Notations:
$$
s_t = x_{t+1} - x_t,\qquad y_t = g_{t+1} - g_t,\qquad y_t = H_{t+1}s_t
$$
- Take $B$ as approximation of $H$, $D$ as approximation of $H^{-1}$:
$$
y_t = B_{t+1}s_t\text{ or }s_t = D_{t+1}y_t
$$
Only when H satisfy the term above, can we apply the Quasi-Newton Method.
### BFGS (Broyden, Fletcher, Goldfarb, Shanno's Method)
The idea is to get close to the Hessian Matrix through iteration.
$$
B_{t+1} = B_t + \Delta B_t,\quad B_0 = E
$$
The initial B is set, hence the key is to construct $\Delta B_t$.

Take $\Delta B_t = \alpha uu^T + \beta vv^T$:
$$
y_t = B_{t+1}s_t = (B_t + \Delta B_t)s_t = B_ts_t + \Delta B_ts_t=B_ts_t+\alpha uu^Ts_t+\beta vv^Ts_t = B_ts_t + u(\alpha u^T s_t) + v(\beta v^Ts_t)
$$
Suppose $\alpha u^Ts_t=1,\beta v^Ts_t=-1$:
$$
\alpha = \frac{1}{u^Ts_t},\qquad \beta = -\frac{1}{v^Ts_t}\\
y_t = B_ts_t + u - v \qquad \Rightarrow \qquad u - v = y_t - B_ts_t
$$
Suppose $u - y_t$, $v = B_ts_t$:
$$
\alpha = \frac{1}{y_t^Ts_t},\qquad \beta = -\frac{1}{(B_ts_t)^Ts_t}\\
\Delta B_t = \frac{y_ty_t^T}{y_t^Ts_t} - \frac{B_ts_t(B_ts_t)^T}{s_t^TB_t^Ts_t}\\
(Sherman-Morrison)D_{t+1} = (E - \frac{s_ty_t^T}{y_t^Ts_t})D_t(E - \frac{y_ts_t^T}{y_t^Ts_t}) + \frac{s_ts_t^T}{y_t^Ts_t}
$$
### In Conclusion
1. $t = 0,D_0=E$;
2. Compute moving direction: $d_t = D_tg_t$;
3. Compute a new $x$: $x_{t+1} = x_t - d_t$;
4. $s_t = d_t$;
5. Check if $||g_{t+1}||\le \epsilon$, decide whether to end the iteration;
6. Compute $y_t = g_{t+1} - g_t$;
7. $t = t + 1$, iteration continues.
### L-BFGS (Limited Memory BFGS)
Alter the original algorithm to make extra space complexity to be constant. 
- Simplify the Hesssian's expression:
$$
\rho_t = \frac{1}{y_t^Ts_t},V_t = E - \rho_ty_ts_t^T\\
D_{t+1} = V_t^TD_tV_t + \rho_ts_ts_t^T
$$
- Use a size-limited queue to store $\rho$s and $V$s.
### In Conclusion
1. Initialize:
$$
\delta = \begin{cases}
    0 & if\quad t\le m\\
    t - m & otherwise
\end{cases},\quad L = \begin{cases}
    t & if\quad t\le m\\
    m & otherwise
\end{cases}
$$
2. Backward iteration:

    for $i=L-1,L-2,...,1,0$:

    $\qquad j = i + \delta$;

    $\qquad a_i = \rho_js_j^Tq_{i+1}$;

    $\qquad q_i = q_{i+1} - \alpha_iy_i$;

3. Forward iteration:

    $r_0 = D_0q_0$;

    for $i=0,...,L-1$:

    $\qquad j=i+\delta$;

    $\qquad\beta_j=\rho_jy_j^Tr_i$;

    $\qquad r_{i+1}=r_i+(\alpha_i-\beta_i)s_i$;

4. $r_L=D_tg_t$

## Better Solution for Logistic Model
### Derivatives
$$
g(w) = \nabla J(w) = \sum_{i=1}^N(\mu_i - y_i)x_i = X^T(\mu - y)\\
H(w) = \frac{\partial g(w)}{\partial w} = \sum_{i=1}^N x_i^T\mu_i(1-\mu_i)x_i = X^TSX,S = diag(\mu_i(1-\mu_i))
$$
### Newton's method
$$
w_{t+1} = w_t - H^{-1}(w_t)g(w) = (X^TS_tX)^{-1}X^TS_tz_t
$$
The formula is similar to Least Square method. Hence, Newton's method in Logistic model is also called Iteratively Reweighted Least Square (IRLS).
## Multi-classifying
### One-vs-Rest(OvR)
Namely, take one of the categories as one category and the rest as another. 
### Multinoulli/Categorical Distribution
Bernoulli distribution has only 2 kinds of output whereas Multinoulli has more kinds of output.
| $X$ | $x_1$ | $x_2$ | $...$ | $x_N$ |
|:-:|:-:|:-:|:-:|:-:|
|$\theta$|$\theta_2$|$\theta_3$|$...$|$\theta_N$|
Write the distribution continuously.
$$
\sum_{c=1}^C\theta_c=1,Cat(y,\theta)=\prod_{c=1}^C\theta^{y_c}_c
$$
Also known as one-hot encoding.
### Softmax
Expend the idea of sigmoid function to be softmax function.
$$
\sigma(z_c) = \frac{e^{z_c}}{\sum_{c'=1}^C e^{z_{c'}}},\mu_c=p(y=c|x,w)=\frac{e^{w_c^Tx}}{\sum_{c'=1}^C e^{w_{c'}^Tx}}
$$
Maximum likelihood estimation:
$$
l(M) = \sum_{i=1}^N\ln\prod_{c=1}^C\mu_{ic}^{y_{ic}}=\sum_{i=1}^N\sum_{c=1}^Cy_{ic}\ln\mu_{ic}
$$
Softmax Loss:
$$
J(w) = -l(M) = \sum_{i=1}^N(\sum_{c=1}^Cy_{ic}w_c^Tx-\ln\sum_{c=1}^Ce^{w_{c'}^Tx})
$$
## Unbalanced Data Sample
Often, some categories give less data sample than other categories. When the difference of the amount of data sample is too significant, we may do something to balance the categories.
### Undersampling
Randomly take a part of the larger sample as a sample, which might cause some damage of information and some error.
### OverSampling
Randomly repeatedly rake a part of the smaller sample as a sample, which might cause the weight of certain parts of the sample to be unjustly magnified and the model to lean upon those parts.
### EasyEnsemble
Generate n samples by n times of under-sampling from larger samples, by which generate n models. The output of the final model is the average of these models.
### BalanceCascade
Generate a sample as large as the small sample by under-sampling from larger samples. Generate a model with this sample and other smaller samples. Select those ones of mistakes and fix the model with other part of the larger sample.
### Synthetic Minority Over-sampling Technique (SMOTE)
- Select a data frame in a smaller sample with index i and vector $x_i$.
- Find K neighbours of $x_i$ in the smaller sample $x_{i(near)},near\in\{1,...,K\}$.
- Randomly select a sample in the neighbours $x_{i(nn)}$. And generate a random number between 0 and 1, $\zeta$,
$$
x_{i1} = (1-\zeta)x_i + \zeta x_{i(near)}
$$
whereas $x_i$ is a point on the straight line between all of the $x_{i(nn)}$ as an insertion.
- SMOTE algorithm can prevent errors from over fitting. However it is not so valid for high dimensional cases and might cause over lap among categories. In order to eliminate the disadvantage of SMOTE, we have Borderline-SMOTE and many other fixing method.
### Cost-Sensitive Learning
Build a cost matrix:
|| Prediction wrong | Prediction right|
|:-:|:-:|:-:|
| Predict 0 | $C_{00}$ | $C_{01}$ |
Predict 1 | $C_{10}$ | $C_{11}$ |
#### Class Weight
1. Regardless of class weights;
2. Balanced mode: Compute class weight according to the amount of certain data samples;
3. Manually set.

## Evaluation
