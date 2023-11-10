from BBV_gradient import *

	
##########################################################################
#************************************************************************#
##########################################################################
class BBV_gradient_Richard_full(BBV_gradient_base):

	def __init__(self, weights, tau):
		BBV_gradient_base.__init__(self)

		self.v0 = 5

		self.T = 60
		self.freq = 30
		self.t = np.linspace(0, self.T, self.T * self.freq, endpoint=False)

		self.wI = 40*weights[0]
		self.wC = 40*weights[1]
		self.a = 10*weights[2]
		self.b = 10*weights[3]
		self.tau_m = weights[4]
		self.sig_m = weights[5]
		self.tau_s = weights[6]
		self.sig_s = weights[7]/100

		self.tp = 0.1
		self.invtau = 1/tau

		self.ncompass = 16
		self.EPG_dirs = np.linspace(0, 2*np.pi, self.ncompass, endpoint=False)

	def simulate(self):
		# Purpose: Simulates movement 
		t = self.t
		dt = t[1] - t[0]

		# initialize arrays
		sol = np.zeros((len(t),3))
		p = np.zeros((len(t)))
		GOAL = np.zeros((len(t),self.ncompass))

		# initialize values
		sol[0,:] = np.array([3.5, 10, 2*np.pi*np.random.rand()])
		p[0] = 1/self.h(self.field(sol[0,0]), self.a,self.b)
		
		epsL = self.ou(self.tau_s, self.sig_s, self.t)
		epsR = self.ou(self.tau_s, self.sig_s, self.t)
		gam = self.ou(self.tau_m, self.sig_m, self.t)

		for i in range(len(t)-1):
			x,y,theta = sol[i,:]
			xyLA = self.get_antL_loc(x,y,theta)
			xyRA = self.get_antR_loc(x,y,theta)

			# Reflect if either antenna hitting wall
			if self.outbounds(xyLA, xyRA):
				x,y,theta = self.reflect(x,y,theta, xyLA, xyRA)
				xyLA = self.get_antL_loc(x,y,theta)
				xyRA = self.get_antR_loc(x,y,theta)
			
			sL = self.field(xyLA[0]) + epsL[i]
			sR = self.field(xyRA[0]) + epsR[i]
			hL, hR = self.h(sL,self.a,self.b), self.h(sR,self.a,self.b)

			hAvg = (hL + hR)/2
			p[i+1] = p[i] + dt*(-p[i] + 1/(hAvg))/self.tp
			d = p[i]*hAvg - 1

			optim = self.EPG_dirs[np.argmax(GOAL[i,:])]
			dtheta = math.remainder(theta - optim, 2*math.pi)

			vL = self.wI * hL + self.wC * hR + self.v0 + gam[i] + dtheta/4
			vR = self.wC * hL + self.wI * hR + self.v0 - gam[i] - dtheta/4

			fx = [0.5*(vL+vR)*np.cos(theta), 0.5*(vL+vR)*np.sin(theta), (vR-vL)/self.d]
			sol[i+1,:] = np.array([x,y,theta]) + dt*np.array(fx)
			EPG = np.cos(self.EPG_dirs-theta)
			GOAL[i+1,:] = GOAL[i,:] + dt*(-self.invtau*GOAL[i,:] - d*EPG)

		return sol
	

		
##########################################################################
#************************************************************************#
##########################################################################
class BBV_gradient_Richard_taxis(BBV_gradient_base):

	def __init__(self, weights, tau):
		BBV_gradient_base.__init__(self)

		self.v0 = 5

		self.T = 60
		self.freq = 30
		self.t = np.linspace(0, self.T, self.T * self.freq, endpoint=False)

		self.wI = 40*weights[0]
		self.wC = 40*weights[1]
		self.a = 10*weights[2]
		self.b = 10*weights[3]
		self.tau_m = weights[4]
		self.sig_m = weights[5]
		self.tau_s = weights[6]
		self.sig_s = weights[7]/100
		self.tp = 0.1

	def simulate(self):
		# Purpose: Simulates movement 
		t = self.t
		dt = t[1] - t[0]

		# initialize arrays
		sol = np.zeros((len(t),3))
		p = np.zeros((len(t)))

		# initialize values
		sol[0,:] = np.array([3.5, 10, 2*np.pi*np.random.rand()])
		p[0] = 1/self.h(self.field(sol[0,0]), self.a,self.b)
		
		epsL = self.ou(self.tau_s, self.sig_s, self.t)
		epsR = self.ou(self.tau_s, self.sig_s, self.t)
		gam = self.ou(self.tau_m, self.sig_m, self.t)

		for i in range(len(t)-1):
			x,y,theta = sol[i,:]
			xyLA = self.get_antL_loc(x,y,theta)
			xyRA = self.get_antR_loc(x,y,theta)

			# Reflect if either antenna hitting wall
			if self.outbounds(xyLA, xyRA):
				x,y,theta = self.reflect(x,y,theta, xyLA, xyRA)
				xyLA = self.get_antL_loc(x,y,theta)
				xyRA = self.get_antR_loc(x,y,theta)
			
			sL = self.field(xyLA[0]) + epsL[i]
			sR = self.field(xyRA[0]) + epsR[i]
			hL, hR = self.h(sL,self.a,self.b), self.h(sR,self.a,self.b)

			hAvg = (hL + hR)/2
			p[i+1] = p[i] + dt*(-p[i] + 1/(hAvg))/self.tp
			d = p[i]*hAvg - 1
			tax = self.h(d, a=20, b=0)
			
			vL = self.wI * hL + self.wC * hR + self.v0 + tax*gam[i]
			vR = self.wC * hL + self.wI * hR + self.v0 - tax*gam[i]

			fx = [0.5*(vL+vR)*np.cos(theta), 0.5*(vL+vR)*np.sin(theta), (vR-vL)/self.d]
			sol[i+1,:] = np.array([x,y,theta]) + dt*np.array(fx)

		return sol