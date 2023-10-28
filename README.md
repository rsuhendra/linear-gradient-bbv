# Original formulation

Original equations are

$$
\left[\begin{array}{c}
x^{\prime} \\
y^{\prime} \\
\theta^{\prime}
\end{array}\right]=\left[\begin{array}{c}
\frac{1}{2}\left(v_L+v_R\right) \cos (\theta) \\
\frac{1}{2}\left(v_L+v_R\right) \sin (\theta) \\
\frac{1}{d}\left(v_R-v_L\right)
\end{array}\right]
$$

$$
\begin{align}
v_L = w_I h(s_L) + w_C h(s_R) + v_0 + \gamma \\ 
v_R = w_C h(s_L) + w_I h(s_R) + v_0 - \gamma 
\end{align}
$$

where $w_I, w_C$ are ipsilateral and contralateral weights, $v_0$ is a base speed, and $\gamma$ is Ornstein-Uhlenbeck colored noise, $h$ is sigmoid function that transforms each sensor input to $[0,1]$.

# Richard additions

## the derivative approximation

## goal direction approximation
We assume $n$ EPG cells. Each cell corresponds to an angle $phi$ distributed evenly across $[0, 2\pi]$, forming a ring which encodes the current heading direction according to
$${EPG}_\phi =  \cos(\phi-\theta)$$

We also assume the direction of another set of $n$ ring cells, which encode the direction of a goal, a preferred heading. These integrate the product of the derivative-like signal $d$ and the EPG cells, and evolve according to 

$$\frac{dG\phi}{dt} = -kG\phi - d\cdot{EPG}_\phi $$

We then take the goal direction at time $t$ to be 
$$ \phi_{opt} = argmax_{\phi} G_\phi(t) $$

## new movement
We retain the equations of motions from (1), but modify the speeds of each vehicle wheel to be

$$
\begin{align}
v_L = v_0 + \gamma - \delta\theta \\ 
v_R = v_0 - \gamma + \delta\theta
\end{align}
$$

where $\delta\theta = \alpha (\theta - \phi_{opt})$. 

Justification for removing the sensor terms is rooted in the observation that there is 


