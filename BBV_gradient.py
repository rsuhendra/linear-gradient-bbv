from BBV import *
from utils import *
import pickle 
from scipy.interpolate import griddata
from scipy import signal
import math
import pandas as pd
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt


(x2,t0) = pickle.load(open("contour_datBig_gradient.pkl","rb"),encoding='latin1')

t0 = t0.squeeze()

# t0 = np.linspace(38,24.5, num=6001)
# x2 = np.linspace(0, 600, num=6001)

##########################################################################
#************************************************************************#
##########################################################################

class BBV_gradient_base(BBV_base):

	def __init__(self):
		BBV_base.__init__(self)

		self.stageW = 350
		self.stageH = 200
		self.epsilon = 5
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
					theta = angle_reflect(theta, reflect_angle)
				else:
					reflect_angle = np.pi/2 + (2*(y>self.stageH/2)-1)*(np.pi/2)
					theta = angle_reflect(theta, reflect_angle)

		theta = math.remainder(theta, 2*math.pi)

		return x, y, theta

	def field(self, x):
		# Purpose: Returns arena temp at given position
		# only depends on x position 
		temp = griddata(x2,t0,x)
	
		return (temp-self.minTemp)/10.
	
	def simulate(self):
		# Purpose: Simulates movement 
		return 0

	def generate_data(self, num_Simulations = 100, output_name='test.output'):

		data = []
		for i in range(num_Simulations):
			sol, vels = self.simulate()
			# _, velsX = smooth_and_deriv(sol[:,0], bbv.dt)
			# _, velsY = smooth_and_deriv(sol[:,1], bbv.dt)
			# _, velsRot = smooth_and_deriv(sol[:,2], bbv.dt)
			# smoothVels = np.vstack((velsX, velsY, velsRot)).T
			data.append([sol, vels])	

		f1 = open('outputs/'+output_name,'wb')
		pickle.dump((self, data),f1)
		f1.close()


##########################################################################
#************************************************************************#
##########################################################################

class BBV_gradient_Josh_simple(BBV_gradient_base):

	def __init__(self, weights):
		BBV_gradient_base.__init__(self)

		self.v0 = 5

		self.T = 60
		self.freq = 15
		self.dt = 1/self.freq
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
		sol[0,:] = np.array([35, 100, 2*np.pi*np.random.rand()])
		
		epsL = self.ou(self.tau_s, self.sig_s, self.t)
		epsR = self.ou(self.tau_s, self.sig_s, self.t)
		gam = self.ou(self.tau_m, self.sig_m, self.t)

		for i in range(len(t)-1):
			x,y,theta = sol[i,:]
			xyLA = self.get_antL_loc(x,y,theta)
			xyRA = self.get_antR_loc(x,y,theta)

			# If past a certain point stop sim
			if x > 0.85*self.stageW:
				return sol[:i,:]		

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

class BBV_gradient_taxis(BBV_gradient_base):

	def __init__(self, weights, p1, p2, mode_taxis = 'perfect', mode_motor = 'ou' ):
		BBV_gradient_base.__init__(self)

		self.v0 = 5

		self.T = 600
		self.freq = 15
		self.dt = 1/self.freq
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

		self.p1 = p1
		self.p2 = p2
		self.mode_motor = mode_motor
		self.mode_taxis = mode_taxis

	def simulate(self):
		# Purpose: Simulates movement 
		t = self.t
		dt = t[1] - t[0]

		# initialize arrays
		sol = np.zeros((len(t),3))
		vels = np.zeros((len(t),3))
		p = np.zeros((len(t)))
		taxes = []

		# initialize values
		sol[0,:] = np.array([35, 100, 2*np.pi*np.random.rand()])
		p[0] = 1/self.h(self.field(sol[0,0]), self.a,self.b)
		
		epsL = self.ou(self.tau_s, self.sig_s, self.t)
		epsR = self.ou(self.tau_s, self.sig_s, self.t)

		if self.mode_motor == 'ou':
			gam = self.ou(self.tau_m, self.sig_m, self.t)
		elif self.mode_motor == 'signed_ou':
			gam = self.ou(self.tau_m, self.sig_m, self.t, mode = 'signed')

		p1, p2 = self.p1, self.p2

		for i in range(len(t)-1):
			x,y,theta = sol[i,:]
			xyLA = self.get_antL_loc(x,y,theta)
			xyRA = self.get_antR_loc(x,y,theta)

			# If past a certain point stop sim
			if x > 0.85*self.stageW:
				return sol[:i,:], vels[:i,:]

			# Reflect if either antenna hitting wall
			if self.outbounds(xyLA, xyRA):
				x,y,theta = self.reflect(x,y,theta, xyLA, xyRA)
				xyLA = self.get_antL_loc(x,y,theta)
				xyRA = self.get_antR_loc(x,y,theta)
			
			sL = self.field(xyLA[0]) + epsL[i]
			sR = self.field(xyRA[0]) + epsR[i]
			hL, hR = self.h(sL,self.a,self.b), self.h(sR,self.a,self.b)
			hAvg = (hL + hR)/2

			if i == 0:
				p[0] = 1/hAvg

			p[i+1] = p[i] + dt*(-p[i] + 1/(hAvg))/self.tp
			d = p[i]*hAvg - 1

			if self.mode_taxis == 'derivative':
				tax = self.h(d, a=200, b=0)
				# print(200*d, tax)
			elif self.mode_taxis == 'perfect':
				tax = angle_diff(0, theta)/np.pi
	
			taxes.append(tax)

			# vL = self.wI * hL + self.wC * hR + self.v0 + p1*tax*gam[i] + p2*tax
			# vR = self.wC * hL + self.wI * hR + self.v0 - tax*gam[i] + 4*tax

			tax_mult = (1-p1) + p1*tax   # p1==0 equates to regular
			vL = self.v0 + tax_mult*gam[i] + p2*(1-tax)
			vR = self.v0 - tax_mult*gam[i] + p2*(1-tax)

			fx = [0.5*(vL+vR)*np.cos(theta), 0.5*(vL+vR)*np.sin(theta), (vR-vL)/self.d]
			fx = np.array(fx)
			if i == 0:
				vels[0,:] = fx
			vels[i+1,:] = fx
			sol[i+1,:] = np.array([x,y,theta]) + dt*np.array(fx)

		return sol, vels

##########################################################################
#************************************************************************#
##########################################################################

class BBV_gradient_taxis_point(BBV_gradient_taxis):

	def __init__(self, weights, p1, p2, brait, mode_taxis = 'perfect', mode_motor = 'ou' ):
		BBV_gradient_taxis.__init__(self, weights, p1, p2, mode_taxis, mode_motor)

		self.v0 = 5

		self.T = 600
		self.freq = 30
		self.dt = 1/self.freq
		self.t = np.linspace(0, self.T, self.T * self.freq, endpoint=False)

		self.brait = brait

	def simulate(self, testing = False):
		# Purpose: Simulates movement 
		t = self.t
		dt = t[1] - t[0]

		# initialize arrays
		sol = np.zeros((len(t),3))
		vels = np.zeros((len(t),3))
		p = np.zeros((len(t)))
		taxes = []

		max_lambda = 0.5
		time_counter = 0
		point_time = np.random.default_rng().exponential(scale=max_lambda)
		pause = False

		# initialize values
		sol[0,:] = np.array([35, 100, 2*np.pi*np.random.rand()])
		p[0] = 1/self.h(self.field(sol[0,0]), self.a,self.b)
		
		epsL = self.ou(self.tau_s, self.sig_s, self.t)
		epsR = self.ou(self.tau_s, self.sig_s, self.t)

		if self.mode_motor == 'ou':
			gam = self.ou(self.tau_m, self.sig_m, self.t)
		elif self.mode_motor == 'signed_ou':
			gam = self.ou(self.tau_m, self.sig_m, self.t, mode = 'signed')

		p1, p2 = self.p1, self.p2

		for i in range(len(t)-1):
			x,y,theta = sol[i,:]
			xyLA = self.get_antL_loc(x,y,theta)
			xyRA = self.get_antR_loc(x,y,theta)

			# If past a certain point stop sim
			if x > 0.85*self.stageW:
				if testing == True:
					return sol[:i,:], vels[:i,:]
				else:
					return sol[:i,:], vels[:i,:]
				
			# Reflect if either antenna hitting wall
			if self.outbounds(xyLA, xyRA):
				x,y,theta = self.reflect(x,y,theta, xyLA, xyRA)
				xyLA = self.get_antL_loc(x,y,theta)
				xyRA = self.get_antR_loc(x,y,theta)
			
			sL = self.field(xyLA[0]) + epsL[i]
			sR = self.field(xyRA[0]) + epsR[i]
			hL, hR = self.h(sL,self.a,self.b), self.h(sR,self.a,self.b)
			# hL = 0 #CHANGE THIS
			hAvg = (hL + hR)/2

			if i == 0:
				p[0] = 1/hAvg

			p[i+1] = p[i] + dt*(-p[i] + 1/(hAvg))/self.tp
			d = p[i]*hAvg - 1

			if self.mode_taxis == 'derivative':
				tax = self.h(d, a=200, b=0)
				# print(200*d, tax)
			elif self.mode_taxis == 'perfect':
				tax = angle_diff(0, theta)/np.pi
	
			taxes.append(tax)

			tax_mult = (1-p1) + p1*tax   # p1==0 equates to regular

			# Find a better way to do this
			if time_counter > point_time:
				time_counter = 0
				pause = True
				point_time = np.random.default_rng().exponential(scale=max_lambda)
				mu, sigma = 2, 0.5 # mean and standard deviation
				s = np.random.lognormal(mu, sigma)	
				test = np.linspace(0, 1, 1+int(s))
				sgn = 2*np.random.randint(2) - 1
				triangle = sgn*(np.pi/2)*(1+signal.sawtooth(2 * np.pi * test,  width = 0.5))

				sgn = self.h(tax_mult, a=10, b=1)
				# print(sgn, theta)
				if np.random.rand()>sgn:
					triangle = np.zeros_like(test)
				timer = len(test)-1
				counter = 0

			vr_add = 0
			if pause == True:
				if timer == 0:
					pause = False
			if pause == True:
				timer -= 1
				counter +=1
				vr_add = triangle[counter]

			vL = self.v0 + vr_add #- tax_mult*gam[i] + p2*(1-tax)
			vR = self.v0 - vr_add #+ tax_mult*gam[i] + p2*(1-tax) 

			if self.brait == True:
				vL += self.wI * hL + self.wC * hR
				vR += self.wC * hL + self.wI * hR

			# add time counter for thing
			if pause == False:
				time_counter += dt

			fx = [0.5*(vL+vR)*np.cos(theta), 0.5*(vL+vR)*np.sin(theta), (vR-vL)/self.d]
			fx = np.array(fx)
			if i == 0:
				vels[0,:] = fx
			vels[i+1,:] = fx
			sol[i+1,:] = np.array([x,y,theta]) + dt*np.array(fx)


		if testing == True:
			return sol, vels
		else:
			return sol, vels

##########################################################################
#************************************************************************#
##########################################################################



class BBV_gradient_taxis_ou_special(BBV_gradient_taxis):

	def __init__(self, weights, p1, p2, mode_taxis = 'perfect', mode_motor = 'ou' ):
		BBV_gradient_taxis.__init__(self, weights, p1, p2, mode_taxis, mode_motor)

	def simulate(self):
		# Purpose: Simulates movement 
		t = self.t
		dt = t[1] - t[0]

		# initialize arrays
		sol = np.zeros((len(t),3))
		vels = np.zeros((len(t),3))
		p = np.zeros((len(t)))
		taxes = []

		# initialize values
		sol[0,:] = np.array([35, 100, 2*np.pi*np.random.rand()])
		p[0] = 1/self.h(self.field(sol[0,0]), self.a,self.b)
		
		epsL = self.ou(self.tau_s, self.sig_s, self.t)
		epsR = self.ou(self.tau_s, self.sig_s, self.t)

		gam = np.zeros((len(t)))

		p1, p2 = self.p1, self.p2

		for i in range(len(t)-1):
			x,y,theta = sol[i,:]
			xyLA = self.get_antL_loc(x,y,theta)
			xyRA = self.get_antR_loc(x,y,theta)

			# If past a certain point stop sim
			if x > 0.85*self.stageW:
				return sol[:i,:], vels[:i,:]

			# Reflect if either antenna hitting wall
			if self.outbounds(xyLA, xyRA):
				x,y,theta = self.reflect(x,y,theta, xyLA, xyRA)
				xyLA = self.get_antL_loc(x,y,theta)
				xyRA = self.get_antR_loc(x,y,theta)
			
			sL = self.field(xyLA[0]) + epsL[i]
			sR = self.field(xyRA[0]) + epsR[i]
			hL, hR = self.h(sL,self.a,self.b), self.h(sR,self.a,self.b)
			hAvg = (hL + hR)/2

			if i == 0:
				p[0] = 1/hAvg

			p[i+1] = p[i] + dt*(-p[i] + 1/(hAvg))/self.tp
			d = p[i]*hAvg - 1

			if self.mode_taxis == 'derivative':
				tax = self.h(d, a=200, b=0)
				# print(200*d, tax)
			elif self.mode_taxis == 'perfect':
				tax = angle_diff(0, theta)/np.pi
	
			taxes.append(tax)

			# vL = self.wI * hL + self.wC * hR + self.v0 + p1*tax*gam[i] + p2*tax
			# vR = self.wC * hL + self.wI * hR + self.v0 - tax*gam[i] + 4*tax

			tax_mult = (1-p1) + p1*tax   # p1==0 equates to regular
			vL = self.v0 + gam[i] + p2*(1-tax)
			vR = self.v0 - gam[i] + p2*(1-tax)

			fx = [0.5*(vL+vR)*np.cos(theta), 0.5*(vL+vR)*np.sin(theta), (vR-vL)/self.d]
			fx = np.array(fx)
			if i == 0:
				vels[0,:] = fx
			vels[i+1,:] = fx
			sol[i+1,:] = np.array([x,y,theta]) + dt*np.array(fx)
			
			gam[i+1] = gam[i] + (-(gam[i]/tax_mult)*dt + self.sig_m*np.sqrt(dt)*np.random.normal())/self.tau_m

		return sol, vels

##########################################################################
#************************************************************************#
##########################################################################
