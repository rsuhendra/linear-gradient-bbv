from BBV import *
import pickle 
from scipy.interpolate import griddata
import math
import pandas as pd
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt

class MemoryBuffer:
	def __init__(self, capacity):
		self.capacity = capacity
		self.buffer = []

	def push(self, x):
		if len(self.buffer) >= self.capacity:
			self.buffer.pop(0)
		self.buffer.append(x)

	def arrayize(self):
		return np.array(self.buffer)

	def __len__(self):
		return len(self.buffer)
	
##########################################################################
#************************************************************************#
##########################################################################

class BBV_gradient_base(BBV_base):

	def __init__(self):
		BBV_base.__init__(self)

		self.stageW = 35
		self.stageH = 20
		self.epsilon = 0.1
		self.minTemp = 24.5
		
		self.name = self.__class__.__name__.split('_', 2)[-1]

	def outbounds(self, xyLA, xyRA, mode = 1):
		# Purpose: Check if antenna in bounds
		LAoutx = xyLA[0]<self.epsilon or xyLA[0]>(self.stageW-self.epsilon)
		LAouty = xyLA[1]<self.epsilon or xyLA[1]>(self.stageH-self.epsilon)
		RAoutx = xyRA[0]<self.epsilon or xyRA[0]>(self.stageW-self.epsilon)
		RAouty = xyRA[1]<self.epsilon or xyRA[1]>(self.stageH-self.epsilon)
		out = LAoutx or LAouty or RAoutx or RAouty
		if mode == 1:
			return out
		elif mode == 2:
			return LAoutx, LAouty, RAoutx, RAouty

	def reflect(self, x,y,theta, xyLA, xyRA):
		# Purpose: reflects the animal when it approaches the boundary
		# for rectangular gradient arena
		LAoutx, LAouty, RAoutx, RAouty = self.outbounds(xyLA, xyRA, mode=2)
		LAout, RAout = LAoutx or LAouty, RAoutx or RAouty

		if LAout and RAout: # if both antenna out, turn around completely
			theta = theta - np.pi
		else:
			if LAout:
				outx, outy = LAoutx, LAouty
				way = 1
			else:
				outx, outy = RAoutx, RAouty
				way = -1
			if outx and outy: #
				theta = theta + way*np.pi/2
			else:
				if outx:
					reflect_angle = np.pi + (2*(x>self.stageW/2)-1)*(np.pi/2)
					theta = self.angle_reflect(theta, reflect_angle)
				else:
					reflect_angle = np.pi/2 + (2*(y>self.stageH/2)-1)*(np.pi/2)
					theta = self.angle_reflect(theta, reflect_angle)

		theta = math.remainder(theta, 2*math.pi)
		return x, y, theta

	def angle_reflect(self,angle, reflect_angle):
		my_angle = angle - 2*(angle - reflect_angle)
		return math.remainder(my_angle, 2*math.pi)
	
	def field(self, x):
		# Purpose: Returns arena temp at given position
		# only depends on x position 
		temp = griddata(x2,t0,x*10)
		return (temp-self.minTemp)/10.
	
	def simulate(self):
		# Purpose: Simulates movement 
		return 0
	
	def sample_plot(self):
		
		sol = self.simulate()

		fig, ax = plt.subplots()
		ax.scatter(sol[:,0], sol[:,1], s=0.5)
		ax.plot(sol[:,0], sol[:,1], linewidth=0.1)
		ax.set(xlim=[0, self.stageW], ylim=[0, self.stageH])
		ax.set_aspect('equal')
		ax.set_title('Sample plot')
		fig.savefig('plots/'+'sample_'+self.name+'.pdf', transparent=True)
		plt.close()

	
	def percent_reached_plot(self, numSimulations=20):
		lineDists = [0.2,0.4,0.6,0.8]
		fig, ax = plt.subplots()
		allLineInds = []
		allCumDists = []

		for _ in range(numSimulations):
			
			sol = self.simulate()
			pos = sol[:,:2]
			normalized_x = sol[:,0]/self.stageW

			lineInds = []
			for l in lineDists:
				ind = next((i for i in range(self.T*self.freq) if normalized_x[i] > l), None)
				lineInds.append(ind if ind is not None else None)

			distances = np.linalg.norm(pos[1:,] - pos[:-1,], axis=1)
			cumulative_distances = np.cumsum(distances)
			lineFirstHitDist = [cumulative_distances[i] if i is not None else None for i in lineInds]
			# check fine details on indices

			allLineInds.append(lineInds)
			allCumDists.append(lineFirstHitDist)

		percent_reached = []
		distances_reached = []
		med_dist_reached = []

		numFiles = len(allLineInds)
		for i in range(len(lineDists)):
			count = 0
			dists = []
			for j in range(numFiles):
				if allLineInds[j][i] is not None:
					count += 1
					dists.append(allCumDists[j][i])

			percent_reached.append(count/numFiles)
			distances_reached.append(dists)
			med_dist_reached.append(np.median(dists))

		ax.boxplot(distances_reached)
		for i,list in enumerate(distances_reached):
			plt.scatter([i+1]*len(list), list)
		ax.set_xticklabels([round(p, 2) for p in percent_reached])
		ax.set_ylabel('Distance walked (cm)')
		ax.set_ylim([0, 300])
		fig.savefig('plots/'+'dist_reached_'+self.name+'.pdf', transparent=True)
		plt.close()

	def speed(self):
		pass
		# velsX = np.gradient(sol[:, 0], self.dt)
		# velsY = np.gradient(sol[:, 1], self.dt)
		

##########################################################################
#************************************************************************#
##########################################################################

class BBV_gradient_Josh_simple(BBV_gradient_base):

	def __init__(self, weights):
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

	def simulate(self):
		# Purpose: Simulates movement 
		t = self.t
		dt = t[1] - t[0]
		sol = np.zeros((len(t),3))
		sol[0,:] = np.array([3.5, 10, 2*np.pi*np.random.rand()])
		
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

			vL = self.wI * hL + self.wC * hR + self.v0 + gam[i]
			vR = self.wC * hL + self.wI * hR + self.v0 - gam[i] 

			fx = [0.5*(vL+vR)*np.cos(theta), 0.5*(vL+vR)*np.sin(theta), (vR-vL)/self.d]
			sol[i+1,:] = np.array([x,y,theta]) + dt*np.array(fx)

		return sol
	
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

			vL = self.wI * hL + self.wC * hR + self.v0 + gam[i] + dtheta/2
			vR = self.wC * hL + self.wI * hR + self.v0 - gam[i] - dtheta/2

			fx = [0.5*(vL+vR)*np.cos(theta), 0.5*(vL+vR)*np.sin(theta), (vR-vL)/self.d]
			sol[i+1,:] = np.array([x,y,theta]) + dt*np.array(fx)
			EPG = np.cos(self.EPG_dirs-theta)
			GOAL[i+1,:] = GOAL[i,:] + dt*(-self.invtau*GOAL[i,:] - d*EPG)

		return sol
		
##########################################################################
#************************************************************************#
##########################################################################

class BBV_gradient_Richard_taxis(BBV_gradient_base):

	def __init__(self, weights):
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




(x2,t0) = pickle.load(open("contour_datBig_gradient.pkl","rb"),encoding='latin1')

t0 = t0.squeeze()

t0 = np.linspace(38,24.5, num=6001)
x2 = np.linspace(0, 600, num=6001)