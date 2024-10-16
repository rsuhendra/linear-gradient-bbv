import numpy as np

class BBV_base(object):
	def __init__(self):
		# fly vehicle parameters (Body Length, Antennal Distance, fly length, base speed in mm)
		self.BL = 3
		self.AD = 0.3
		self.d = 0.75

	def get_antL_loc(self, x, y, theta):
		# Purpose: returns left antennal locations
		return x + self.BL/2*np.cos(theta) - self.AD/2*np.sin(theta), y + self.BL/2*np.sin(theta) + self.AD/2*np.cos(theta)

	def get_antR_loc(self, x, y, theta):
		# Purpose: returns right antennal locations
		return x + self.BL/2*np.cos(theta) + self.AD/2*np.sin(theta), y + self.BL/2*np.sin(theta) - self.AD/2*np.cos(theta)

	def get_head_loc(self, x, y, theta):
		# Purpose: returns head locations (UNUSED)
		return x + self.BL/2*np.cos(theta), y + self.BL/2*np.sin(theta)
	
	def ou(self, tau, sig, t, y0=0, mode = 'regular'):
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
	
	def h(self, s,a,b):
		# Purpose: Sigmoid function
		return 1./(1+np.exp(-a*s+b))
	
	def line_projection(self, x, y, theta, d=1):
		# Projection direction theta to line in front of dot, for plotting
		# d = length of projections
		return np.array([x, x+d*np.cos(theta)]), np.array([y, y+d*np.sin(theta)])
	
	def simulate(self):
		# Purpose: Simulates movement 
		raise NotImplementedError("Subclasses should implement this!")
