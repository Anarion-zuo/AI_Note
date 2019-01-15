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
The Gauss Module is symmetric. Nearer it is to the center, larger the number is, which reminds us of the Gauss distribution.
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
