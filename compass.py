import numpy as np
import matplotlib.pyplot as plt
from utils import *

def ou(tau, sig, t, y0=0, mode = 'regular'):
		# Purpose: Ornstein-Uhlenbeck noise
		dt=t[1]-t[0]
		sqrtdt = np.sqrt(dt)
		y=np.zeros(t.shape)
		y[0]=y0
		wt=sqrtdt*np.random.normal(size=t.shape)
		for i in range(len(t)-1):
			if mode == 'regular':
				y[i+1]=y[i] + (-y[i]*dt + sig*wt[i])/tau
			elif mode == 'signed':
				y[i+1]=y[i] + (-np.sign(y[i])*dt + sig*wt[i])/tau
		return y

# Function for ringrhs
def ringrhs(v, Wmat, tau, g, N, input):
	recurr = (2 * g / N) * np.dot(Wmat, v)
	nonlin = ReLU(recurr + input)
	dydt = (-v + nonlin) / tau
	return dydt


def simulate(Wmat, N = 12, g = 3, tau = 1, epsilon = 1, amp = 5.0, thetaG = np.pi/2, tfinal = 500, dt = 0.01, tau_m = 0.65, sig_m = 0.45, input_reverse = False):

	# Initialization step
	x = np.linspace(-np.pi, np.pi, N, endpoint=False).reshape(-1, 1)

	time = np.linspace(0, tfinal, num = int(tfinal/dt)+1)

	# Randomize initial direction
	th = np.random.uniform(low=-np.pi, high=np.pi)
	headings = [th]

	angvel = ou(tau_m, sig_m, time)

	# initialize v
	v = 10 * (1.0 + epsilon * (np.cos(x - np.pi)))
	# v = np.zeros_like(x)


	goals = [calc_direction(v,x)]
	goalv = [v[:,0]]

	# Timestepping
	for i in range(len(time)-1):
		# input = amp * (1.0 + epsilon * (np.cos(x - th))) * ReLU(np.cos(th - thetaG))
		# if input_reverse == True:
		# 	input += amp * (1.0 + epsilon * (np.cos(x - th - np.pi))) * ReLU(-np.cos(th - thetaG))

		input = amp * (epsilon * (np.cos(x - th)) + 5*np.cos(th - thetaG))
  
		v = v + dt * ringrhs(v, Wmat, tau, g, N, input)
		th = th + dt*angvel[i]
		
		headings.append(th)
		goals.append(calc_direction(v,x))
		goalv.append(v[:,0])

	headings = cast_to(np.array(headings))
	goals = np.array(goals)
	goalv = np.array(goalv)

	return time, headings, goals, goalv

# # Parameters
# N = 12
# dt = 0.01
# thetaG = np.pi/2

# # Construct W matrix
# Wrow = np.cos(np.arange(N) * 2 * np.pi / N)
# Wrow = ReLU(Wrow)
# # Wrow = np.zeros_like(Wrow)
# Wrow[0] = -1
# # Wrow[1] = 1
# # Wrow[-1] = 1
# Wmat = np.array([np.roll(Wrow, n) for n in range(N)])


# time, headings, goals, levels = simulate(Wmat = Wmat, tau=5)





# Gs = [1, 2, 4]
# data = []
# for g in Gs:
# 	devs = []
# 	for i in range(20):
# 		headings, goals, levels = simulate(Wmat = Wmat, tau=g)
# 		skip = int(50/dt)
# 		dev = np.average(np.sin(goals[skip:]-thetaG)**2)
# 		devs.append(dev)
# 	data.append(devs)

# plt.boxplot(data)
# plt.show()
# plt.close()
