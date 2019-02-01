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
In Scikit-Learn, the constant \lambda is differently placed.
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
### Quasi-Newton Method