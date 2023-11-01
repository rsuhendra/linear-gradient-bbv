# Original formulation

Original equations are

$$
\left[\begin{array}{c}
\dot{x} \\
\dot{y} \\
\dot{\theta}
\end{array}\right]=\left[\begin{array}{c}
\frac{1}{2}\left(v_L+v_R\right) \cos (\theta) \\
\frac{1}{2}\left(v_L+v_R\right) \sin (\theta) \\
\frac{1}{d}\left(v_R-v_L\right)
\end{array}\right] \tag{1}
$$

$$
\begin{align} \tag{2}
v_L = w_I h(s_L) + w_C h(s_R) + v_0 + \gamma \\ 
v_R = w_C h(s_L) + w_I h(s_R) + v_0 - \gamma 
\end{align}
$$

where $w_I, w_C$ are ipsilateral and contralateral weights, $v_0$ is a base speed, and $\gamma$ is Ornstein-Uhlenbeck colored noise, $h$ is sigmoid function that transforms each sensor input to $[0,1]$.

# Richard additions

## the derivative approximation
Starting from this basic formulation, we designed a simple filter inspired by a firing-rate model, constructed by modifying a more detailed model of an adapting synapse. For the simplified model we assume an adaptation variable $p$ that evolves according to

$$ \tau_p \frac{dp}{dt} = \frac{r_0}{r} - p $$

where $r$ is the firing rate of the input, and the overall output signal is $d = d_0pr$. This adaptation model clearly needs modification at small input firing rates, but since here the input rate never becomes zero this detail can be safely ignored. This model possesses a number of properties that made it desirable for detecting changes in the input. In steady state (where we assume constant input $r$), $p = \frac{r_0}{r}$ and thus $d = pr = r_0 d_0$, which is independent of the input $r$. Since we will add an additional weight to this signal, we will take $r_0 = d_0 = 1$ without loss of generality. Another key property is the transient response of $d$ to a fast-changing input. If the input is a step-function from a smaller input $r_1$ to a larger input $r_2$, since the value of $p$ takes time $\tau_p$ to change appreciably, the immediate change in $g$ follows $\Delta g = p \Delta r = \frac{r_2-r_1}{r_1}$, i.e., the well-known Weber-Fechner law.

In this example, we take the $r = \frac{h(s_L)+h(s_R)}{2}$, and use $d\rightarrow d-1$ to achieve the desired parity.


## goal direction approximation
We assume $n$ EPG cells. Each cell corresponds to an angle $\phi$ distributed evenly across $[0, 2\pi]$, forming a ring which encodes the current heading direction according to
$${EPG}_\phi =  \cos(\theta-\phi)$$

We also assume the direction of another set of $n$ ring cells, which encode the direction of a goal, a preferred heading. These integrate the product of the derivative-like signal $d$ and the EPG cells, and evolve according to 

$$\frac{dG_\phi}{dt} = -kG_\phi - d\cdot {EPG}_\phi \tag{3} $$

We then take the goal direction at time $t$ to be 
$$\hat{\phi} = \arg\max_{\phi} G_\phi$$

## new movement
We retain the equations of motions from (1), but modify the speeds of each vehicle wheel to be

$$
\begin{align} \tag{4}
v_L &= v_0 + \gamma - \delta\theta \\ 
v_R &= v_0 - \gamma + \delta\theta 
\end{align}
$$

where $\delta\theta = \frac{\alpha}{2} (\hat{\phi}-\theta)$ is a corrective term with gain $\alpha$

Justification for removing the sensor terms is rooted in the observation that the temperature difference between the two antennae is impossible to detect in the linear regime. Therefore, the sensor terms should not result in differential speeds between the two motors. This is ignoring that overall speed is higher when at higher temperatures (unless $w_I$ and $w_C$ are equal which they are not in most cases), but I chose to ignore this interaction for now.

# How does G compute the goal direction

Firstly, we assume that the gradient is linear, and varies in only one direction. In our situation, the optimal direction is $\phi_{opt}=0$ therefore the gradient vector is $\Delta f = <-a,0>$. Therefore, the directional derivative for movement in some direction $\theta$ is going to be $-a\cos(\theta-\phi_{opt})$. Generalizing to arbitrary $\phi_{opt}$ . Assuming a constant speed, we then have $\frac{dTemp}{dt}(\theta) = -av_0 \cos(\theta-\phi_{opt})$. Since we assume $d\approx b\frac{dTemp}{dt}$, we have 

$$
\begin{align}
\frac{dG_\phi}{dt} &= -kG_\phi - d\cdot EPG_\phi \\ 
&\approx -kG_\phi + \beta\cos(\theta-\phi_{opt}) \cdot \cos(\theta-\phi) \tag{5}
\end{align}
$$

Looking at steady states and assuming we are travelling only in a direction $\theta_0$, we get

$$
G_\phi  = \frac{\beta}{k} \cos(\theta_0-\phi_{opt}) \cdot \cos(\theta_0-\phi)
$$

which implies that $\hat{\phi}= \theta_0$ if $\cos(\theta_0-\phi_{opt})>0$ or $\hat{\phi}= \theta_0-\pi$ if $\cos(\theta_0-\phi_{opt})<0$. This implies that computed goal direction is always at least within $\frac{\pi}{2}$ of the optimal direction $\phi_{opt}$ and that it is always a direction of descent. 

The process by which the $G$ computes the optimal direction is by sampling. Since the term is weighted by $\cos(\theta_0-\phi_{opt})$, travelling in random directions tends to yield the optimal travel direction. 


# How the corrective term makes the vehicle go in the goal direction
Since we have 

$$ 
\begin{align}
\dot{\theta} &= \frac{1}{d}\left(v_R-v_L\right) \\
&= \frac{1}{d}\left( -2\gamma + 2\delta\theta  \right) \\
&= \frac{1}{d}\left( -2\gamma + \alpha(\hat{\phi}-\theta)\right)
\end{align}
$$

In the absence of the $\gamma$ term which has mean zero, we get 
$$\dot{\theta} = \frac{\alpha}{d}\left(\hat{\phi}-\theta\right)$$
Implying that $\theta$ corrects to $\hat{\phi}$ with the in the presence of $\delta\theta$

# What those videos are
