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
