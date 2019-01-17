#Filtering
We usually deal with some time-based signal, while when dealing with images, we have garph-based signal. Similarly, we apply the convolution method here.
$$
f(x,y) * g(x,y) = \frac{1}{NM} \sum^{N-1}_{n=0} \sum^{M-1}_{m=0} f(n,m)g(x-n,y-n)
$$
It can also be written in this way.
$$
f(x,y) * g(x,y) = \frac{1}{NM} \sum^{N-1}_{n=0} \sum^{M-1}_{m=0} f(n+1,m+1)g(x,y)
$$
The 2 ways of expression are equivalent for a symmatric filter.

## Filtering on One Pixel
When we try to apply a filter upon a certain pixel, we follow the following steps.
1. Select or build a filter of $3\times 3$ matrix.
2. Fit the filter onto the left-up most place of the image.
3. Compute the product of each pixel having the same position.
4. Add the products to the central pixel.

## Mean Filtering
Simplest case:
$$
g(x,y) = \frac{1}{M} \sum_{(m,n)\in S} f(m,n)
$$
$M$ is for setting the luminance to be the same. $S$ can be:
$$
\begin{gather*}
\frac{1}{5}
\begin{pmatrix}
0 & 1 & 0\\
1 & 1 & 1\\
0 & 1 & 0\\
\end{pmatrix}
\qquad \frac{1}{8}
\begin{pmatrix}
1 & 1 & 1\\
1 & 0 & 1\\
1 & 1 & 1\\
\end{pmatrix}
\qquad \frac{1}{9}
\begin{pmatrix}
1 & 1 & 1\\
1 & 1 & 1\\
1 & 1 & 1\\
\end{pmatrix}
\end{gather*}
$$
Weighted average:
$$
g(x,y) = \frac{1}{M} \sum_{(m,n)\in S} w_{mn}f(m,n)
$$
Specially, Gauss Template:
$$
\begin{gather*}
\frac{1}{10}
\begin{pmatrix}
1 & 1 & 1\\
1 & 2 & 1\\
1 & 1 & 1\\
\end{pmatrix}
\qquad \frac{1}{16}
\begin{pmatrix}
2 & 2 & 2\\
2 & 4 & 2\\
2 & 2 & 2\\
\end{pmatrix}
\end{gather*}
$$
The Gauss Module is symmetric. Nearer it is to the center, larger the number is, because it follows the Gauss distribution upon X and Y axis.
## Median Filtering
For some noise called the Salt-and pepper Noise, the median filter has a good effect. It follows the following steps:
1. Pick some pixels in the filter box. Usually like:
$$
\begin{gather*}
\begin{pmatrix}
\bullet &  &  \\
\bullet & \bullet & \bullet\\
\bullet &   &  \\
\end{pmatrix}
\qquad
\begin{pmatrix}
 & \bullet &  \\
\bullet & \bullet & \bullet\\
 & \bullet  &  \\
\end{pmatrix}
\begin{pmatrix}
\bullet & \bullet & \bullet \\
\bullet & \bullet & \bullet \\
\bullet & \bullet & \bullet \\
\end{pmatrix}\\
\begin{pmatrix}
& & \bullet & & \\
& & \bullet & & \\
\bullet & \bullet &\bullet &\bullet & \bullet\\
& & \bullet & & \\
& & \bullet & & \\
\end{pmatrix}
\begin{pmatrix}
& & \bullet & & \\
& \bullet & \bullet & \bullet & \\
\bullet & \bullet &\bullet &\bullet & \bullet\\
& \bullet & \bullet & \bullet & \\
& & \bullet & & \\
\end{pmatrix}
\end{gather*}
$$
2. Find the mdian value of the selected pixels. Set it to be the output.

## Dilation and Erosion
Notation:
$$
A \oplus B = \bigcup_{b \in B} (A)_{b}\\
A \Theta B = \bigcap_{b \in B} (A)_{-b}
$$
We have our original image and a structure operator, and we move the original picture according to the structure operator. The 2 operator can combine with each other and become a new kind of operation.

Opening Operation:
$$
A \circ B = (A \Theta B) \oplus B
$$
Closing Operation:
$$
A \bullet B = (A \oplus B)\Theta B
$$
Apply Opening then Closing, the operation can effectively firlt all kinds of noises.

## OpenCV
    //高斯滤波
    void GaussianBlur( InputArray src, OutputArray dst, Size ksize, double sigmaX, double sigmaY = 0, int borderType = BORDER_DEFAULT );

    //中值滤波
    void medianBlur( InputArray src, OutputArray dst, int ksize );

    //形态学滤波
    void morphologyEx( InputArray src, OutputArray dst, int op, InputArray kernel, Point anchor=Point(-1,-1), int iterations=1, int borderType=BORDER_CONSTANT, const Scalar& borderValue=morphologyDefaultBorderValue() );
    //op: MORPH_OPEN – 开运算; MORPH_CLOSE – 闭运算 等
# Boundary
There is lots of information contained inside the boundary of an image. In fact, most of the information we get from an image comes from boundaries. To find the boudaries inside an image, we take the derivative of the image to find the maximun value of the derivative, where the color of the image changes rapidly. Therefore, some kind of boundary can be found.
## Robert Operator
$$
\begin{gather*}
x-axis:\quad \begin{pmatrix}
1 & 0 \\ 0 & -1 \\
\end{pmatrix}\qquad
y-axis:\quad \begin{pmatrix}
0 & 1 \\ -1 & 0 \\
\end{pmatrix}
\end{gather*}
$$
To write the formular in a mathmatical form:
$$
E_x = \frac{\partial f(x,y)}{\partial x} = f(x,y)-f(x-1,y-1)\\
E_y = \frac{\partial f(x,y)}{\partial x} = f(x-1,y)-f(x,y-1)
$$
Simply, we take the partial derivative at each position of the image.
## Sobel Operator
A better way to compute partial derivative.
$$
x-axis:
\begin{pmatrix}
-1 & -2 & -1 \\
0 & 0 & 0 \\
1 & 2 & 1 \\
\end{pmatrix} \quad
y-axis:
\begin{pmatrix}
-1 & 0 & 1 \\
-2 & 0 & 2 \\
-1 & 0 & 1 \\
\end{pmatrix}
$$
$$
\frac{\partial f(x,y)}{\partial x} = [f(x-1,y+1)+2f(x,y+1)+f(x+1,y+1)]-[f(x-1,y-1)+2f(x,y-1)+f(x+1,y-1)]\\
\frac{\partial f(x,y)}{\partial y} = [f(x-1,y-1)+2f(x-1,y)+f(x-1,y+1)]-[f(x+1,y-1)+2f(x+1,y)+f(x+1,y+1)]
$$
To take the result as a vector:
$$
M(x,y) = \sqrt {E_{x}^2(x,y) + E^2_y(x,y)}\\
\theta (x,y) = tan^{-1}\frac{E_y(x,y)}{E_x(x,y)}
$$
## Laplace Operator
By taking the first derivative, we find the maximun value thereby find the boundary. It is also a valid method that we take the second derivative thereby find the zero of the second derivative, thereby find the critical point of the first derivative.
$$
\Delta f(x,y) = \nabla ^2 f(x,y) = \frac{\partial ^2 f}{\partial x^2} + \frac{\partial ^2 f}{\partial y^2}\\
= f(x+1,y) + f(x-1,y) + f(x,y+1) + f(x,y-1) - 4f(x,y)\\
\begin{pmatrix}
0 & 1 & 0\\
1 & -4 & 1\\
0 & 1 & 0\\
\end{pmatrix}\qquad
\begin{pmatrix}
1 & 1 & 1\\
1 & -8 & 1\\
1 & 1 & 1\\
\end{pmatrix}
$$
## LoG Operator
It is obvious that the operator taking the second derivative might be too sensitive towards noises. Therefore, we have a new operator which does not have the disadvantage of the Laplace operator and have the advantage of it. We first apply a Gauss Filter to the image then apply the Laplace Operator.
$$
\Delta [G_\sigma (x,y) * f(x,y)] = [\Delta G_\sigma (x,y)] * f(x,y) = LoG * f(x,y)\\
\begin{pmatrix}
-2 & -4 & -4 & -4 & -2\\
-4 & 0 & 8 & 0 & -4\\
-4 & 8 & 24 & 8 & -4\\
-4 & 0 & 8 & 0 & -4\\
-2 & -4 & -4 & -4 & -2\\
\end{pmatrix}
$$
## Canny Operator
An operator that could get continous boundary curve.
1. Filt and Differentiate
$$
H(x,y) = \nabla (G(x,y) * f(x,y)) = \nabla G(x,y) * f(x,y)\\
\nabla G = \begin{pmatrix}\frac{\partial G}{\partial x} \\ \frac{\partial G}{\partial y}\end{pmatrix}\qquad G(x,y) = \frac{1}{\sqrt {2\pi} \sigma}e^{-\frac{x^2+y^2}{\sigma ^2}}
$$
2. Compute Gradient
