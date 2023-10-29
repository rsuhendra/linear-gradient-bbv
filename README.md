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
$${EPG}_\phi =  \cos(\theta-\phi)$$

We also assume the direction of another set of $n$ ring cells, which encode the direction of a goal, a preferred heading. These integrate the product of the derivative-like signal $d$ and the EPG cells, and evolve according to 

$$\frac{dG_\phi}{dt} = -kG_\phi - d\cdot {EPG}_\phi $$

We then take the goal direction at time $t$ to be 
$$\phi_{opt} = \arg\max_{\phi} G_\phi$$

## new movement
We retain the equations of motions from (1), but modify the speeds of each vehicle wheel to be

$$
\begin{align}
v_L &= v_0 + \gamma - \delta\theta \\ 
v_R &= v_0 - \gamma + \delta\theta
\end{align}
$$

where $\delta\theta = \alpha (\phi_{opt}-\theta)$. 

Justification for removing the sensor terms is rooted in the observation that the temperature difference between the two antennae is impossible to detect in the linear regime. Therefore, the sensor terms should not result in differential speeds between the two motors. This is ignoring that overall speed is higher when at higher temperatures, but overall this doesn't matter very much.

# How does G compute the goal direction

Firstly, we assume that the gradient is linear, and varies in only one direction. In our situation, the optimal direction is $\phi_{opt}=0$ therefore the gradient vector is $\Delta f = <-a,0>$. Therefore, the directional derivative for movement in some direction $\theta$ is going to be $-a\cos(\theta-\phi_{opt})$. Generalizing to arbitrary $\phi_{opt}$ . Assuming a constant speed, we then have $\frac{dTemp}{dt}(\theta) = -av_0 \cos(\theta-\phi_{opt})$ . Since we assume $d\approx b\frac{dTemp}{dt}$, we have 

$$
\begin{align}
\frac{dG_\phi}{dt} &= -kG_\phi - d\cdot EPG_\phi \\
&= -kG_\phi - \beta\cos(\theta-\phi_{opt}) \cdot \cos(\theta-\phi)
\end{align}
$$

# How the corrective term makes the vehicle go in the goal direction
Since we have 

$$ 
\begin{align}
\theta^{\prime} &= \frac{1}{d}\left(v_R-v_L\right) \\
&= \frac{1}{d}\left( -2\gamma + 2\delta\theta  \right) \\
&= \frac{1}{d}\left( -2\gamma + 2\alpha(\phi_{opt}-\theta)\right)
\end{align}
$$

In the absence of the $\gamma$ term which has mean zero, we get 
$$\theta^{\prime} = \frac{2\alpha}{d}\left(\phi_{opt}-\theta\right)$$
Implying that $\theta$ corrects to $\phi_{opt}$ with the corrective term
