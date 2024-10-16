import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from compass import *

# Parameters
N = 12
thetaG = 0
dt = 1/30
g = 1.75
tau = 5
input_reverse = True

# Construct Wrow

# # WTA
# g = N/2
# K0, K1 = 0.8, -0.5
# Wrow = K1*np.ones(N)
# Wrow[0] = K0

# Cosine
g = 1.75
Wrow = np.cos(np.arange(N) * 2 * np.pi / N)

# # Local model
# g = 2.5
# alpha, beta, D = 2, 1.0, 1
# Wrow = -beta*np.ones(N)
# Wrow[:2] = Wrow[:2] + [alpha - 2*D, D]
# Wrow[-1] = Wrow[-1] + D

# # Local Cosine
# alpha, beta, D = 2.0, 1.0, 0.3
# # alpha, beta, D = 2.0, 0.5, 1.0
# Wrow = D*np.cos(np.arange(N) * 2 * np.pi / N)
# Wrow = ReLU(Wrow) - beta
# Wrow[0] += alpha

# Construct Wmat
Wmat = np.array([np.roll(Wrow, n) for n in range(N)])

# Look at eigenvals
eig, evec = np.linalg.eigh(Wmat)
print('Largest eigenvalue:', (2 * g / N)*np.max(eig))

# simulate
time, headings, goals, goalv = simulate(Wmat = Wmat, N = N, thetaG = thetaG, dt=dt, tau=tau, g=g, input_reverse=input_reverse, amp = 5)

# START ANIMATING
thetas_toplot = np.linspace(-np.pi, np.pi, N+1)

# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

fig = plt.figure()
ax1 = plt.subplot(211, projection='polar')
ax2 = plt.subplot(212)

# fig, (ax1, ax2) = plt.subplots(2, 1, subplot_kw={})
# ax1 = plt.subplot(211, projection='polar')
# ax2 = plt.subplot(212)

# plotting goal vector
# padding so that the plot wraps
goals_plot = np.zeros((goalv.shape[0], goalv.shape[1]+1))
goals_plot[:,:-1] = goalv
goals_plot[:,-1] = goalv[:,0]

polplot, = ax1.plot(thetas_toplot, goals_plot[0,:], label='Goal population')
ax1.set(ylim=[np.min(goalv), np.max(goalv)])

# Plot thetaGoal
ax1.axvline(thetaG, color='green', label='Best direction')

# plotting GOAL
polplot2 = ax1.axvline(goals[0], color='purple', label='Goal direction')

# plotting EPG
polplot3 = ax1.axvline(headings[0], color='red', label = 'Heading direction')

# Plotting everything
ax2.plot(time, headings, color='red')
ax2.plot(time, goals, color='purple')
ax2.axhline(thetaG, color='green')
polplot4 = ax2.axvline(time[0], color='black')

ax2.set_title('Time history')

ax1.legend(loc='upper right', bbox_to_anchor=(2.3, 1.0))

# ax2.legend()

fig.tight_layout()

def update(fr):

	frame = fr + int(0/dt)

	# update polar plot
	polplot.set_ydata(goals_plot[frame,:])

	# update GOAL plot
	polplot2.set_xdata([goals[frame], goals[frame]])

	# update EPG plot
	polplot3.set_xdata([headings[frame], headings[frame]])

	# update EPG plot
	polplot4.set_xdata([time[frame], time[frame]])

ani = animation.FuncAnimation(fig=fig, func=update, frames=180, interval=30)
writervideo = animation.FFMpegWriter(fps=30) 
ani.save('test_add.mp4', writer=writervideo) 
plt.show()


