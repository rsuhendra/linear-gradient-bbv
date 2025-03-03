# Linear Gradient Braitenberg Vehicle (BBV)


## Original formulation

Simões, J.M., Levy, J.I., Zaharieva, E.E. et al. Robustness and plasticity in Drosophila heat avoidance. Nat Commun 12, 2044 (2021). https://doi.org/10.1038/s41467-021-22322-w

The vehicle follows the dynamics

$$
\begin{bmatrix}
\dot{x} \\
\dot{y} \\
\dot{\theta}
\end{bmatrix} =
\begin{bmatrix}
\frac{1}{2}(v_L+v_R) \cos (\theta) \\
\frac{1}{2}(v_L+v_R) \sin (\theta) \\
\frac{1}{d} (v_R-v_L)
\end{bmatrix}
$$

The left and right wheel speeds $v_L, v_R$ follow the equations:

$$
\begin{align}
 v_L &= w_I h(s_L) + w_C h(s_R) + v_0 + \gamma \\
 v_R &= w_C h(s_L) + w_I h(s_R) + v_0 - \gamma
\end{align}
$$

where $s_L, s_R$ are temperatures at the left and right sensor, $w_I, w_C$ are ipsilateral and contralateral weights, $v_0$ is a base speed, and $\gamma$ is Ornstein-Uhlenbeck colored noise. The function $h$ is a sigmoid transformation mapping each sensor input to $[0,1]$.

# Richard's Additions

## Derivative Approximation
We designed a simple filter inspired by a firing-rate model, constructed by modifying a more detailed model of an adapting synapse. For the simplified model, we assume an adaptation variable $p$ that evolves according to

$$ \tau_p \frac{dp}{dt} = \frac{r_0}{r} - p $$

where $r$ is the firing rate of the input, and the overall output signal is $d = d_0pr$. This adaptation model requires modification at small input firing rates, but since the input rate never becomes zero, this detail can be ignored. This model possesses desirable properties for detecting changes in the input. In steady state (assuming constant input $r$), we have

$$ p = \frac{r_0}{r}, \quad d = pr = r_0 d_0 $$

which is independent of $r$. Since an additional weight will be added to this signal, we take $r_0 = d_0 = 1$ without loss of generality.

Another key property is the transient response of $d$ to a fast-changing input. If the input steps from $r_1$ to $r_2$, the immediate change in $d$ follows

$$ \Delta d = p \Delta r = \frac{r_2-r_1}{r_1}, $$

i.e., the well-known Weber-Fechner law.

For this example, we take $r = \frac{h(s_L)+h(s_R)}{2}$ and use $d \to d-1$ to achieve the desired parity.

## Goal Direction Approximation
We assume $n$ EPG cells. Each cell corresponds to an angle $\phi$ distributed evenly across $[0, 2\pi]$, forming a ring that encodes the current heading direction as:

$$ {EPG}_\phi =  \cos(\theta - \phi) $$

We also assume another set of $n$ ring cells encoding the direction of a goal (preferred heading). These integrate the product of the derivative-like signal $d$ and the EPG cells, evolving as:

$$ \frac{dG_\phi}{dt} = -kG_\phi - d \cdot {EPG}_\phi $$

The goal direction at time $t$ is then given by:

$$ \hat{\phi}(t) = \arg\max_{\phi} G_\phi(t) $$

## New Movement
We retain the equations of motion from (1) but modify the speeds of each vehicle wheel:

$$
\begin{align}
 v_L &= v_0 + \gamma - \delta\theta \\
 v_R &= v_0 - \gamma + \delta\theta
\end{align}
$$

where $\delta\theta = \frac{\alpha}{2} (\hat{\phi} - \theta)$ is a corrective term with gain $\alpha$.

The sensor terms were removed because the temperature difference between the two antennae is negligible in the linear regime. Although higher temperatures generally increase overall speed (unless $w_I = w_C$), this interaction was ignored for simplicity.

## How $G$ Computes the Goal Direction

Assuming a linear temperature gradient varying in one direction with $\phi_{opt} = 0$, the gradient vector is $\nabla f = \langle -a, 0 \rangle$. The directional derivative for movement in direction $\theta$ is:

$$ \frac{d\text{Temp}}{dt}(\theta) = -a v_0 \cos(\theta - \phi_{opt}) $$

Approximating $d$ as $d \approx b \frac{d\text{Temp}}{dt}$, we obtain:

$$
\begin{align}
 \frac{dG_\phi}{dt} &= -kG_\phi - d \cdot EPG_\phi \\
 &\approx -kG_\phi + \beta \cos(\theta - \phi_{opt}) \cos(\theta - \phi)
\end{align}
$$

At steady state, assuming motion in direction $\theta_0$:

$$ G_\phi  = \frac{\beta}{k} \cos(\theta_0 - \phi_{opt}) \cos(\theta_0 - \phi) $$

This implies:

- $\hat{\phi} = \theta_0$ if $\cos(\theta_0 - \phi_{opt}) > 0$
- $\hat{\phi} = \theta_0 - \pi$ otherwise.

Thus, the computed goal direction is always within $\frac{\pi}{2}$ of the optimal direction $\phi_{opt}$, ensuring descent.

## How the Corrective Term Aligns the Vehicle

From equation (1), we derive:

$$
\begin{align}
 \dot{\theta} &= \frac{1}{d} (v_R - v_L) \\
 &= \frac{1}{d} (-2\gamma + 2\delta\theta) \\
 &= \frac{1}{d} (-2\gamma + \alpha (\hat{\phi} - \theta))
\end{align}
$$

Ignoring the noise term ($\gamma$ has mean zero), we get:

$$ \dot{\theta} = \frac{\alpha}{d} (\hat{\phi} - \theta) $$

showing that $\theta$ corrects toward $\hat{\phi}$.

# Video Descriptions

## `simul_withbounds` Directory
Equation (3) is implemented. Four videos depict modifications to equation (4):
- "Correction" refers to the $\delta\theta$ term.
- "Noise" refers to the $\gamma$ term.
- The goal is computed as in equation (3) but is not followed when correction is off.

GIF versions are in the `gif` folder.

## `simul_nobounds` Directory
An unbounded version of equation (5) is implemented with modified equation (4), no noise, and varied $\alpha$. A time-delay version simulates real-life conditions.

## `simul_fakebounds` Directory
Similar to `simul_withbounds`, with no noise and extended arena bounds. Labels 1, 2, 3 indicate different angle changes.

