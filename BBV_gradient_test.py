from BBV_gradient import *

##########################################################################
#************************************************************************#
##########################################################################

class BBV_gradient_Richard_test(BBV_gradient_base):

	def __init__(self, weights, tau, mode=1):
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
		
		self.ncompass = 100
		self.EPG_dirs = np.linspace(0, 2*np.pi, self.ncompass, endpoint=False)

		self.mode = mode

	def simulate(self):
		# Purpose: Simulates movement 
		t = self.t
		dt = t[1] - t[0]
	
		# initialize arrays
		sol = np.zeros((len(t),3))
		p = np.zeros((len(t)))
		GOAL = np.zeros((len(t),self.ncompass))

		# checking
		if self.mode == 1:
			optims = np.zeros((len(t)))
			optim_count = np.zeros(self.ncompass)
		elif self.mode == 2:
			pass

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
			GOAL[i+1,:][GOAL[i+1,:]<0]=0

			if self.mode == 1:
				optims[i+1] = self.EPG_dirs[np.argmax(GOAL[i+1,:])]
				optim_count[np.argmax(GOAL[i+1,:])] +=1
			elif self.mode == 2:
				pass
		
		if self.mode == 1:
			return sol, optim, optim_count, GOAL
		elif self.mode == 2:
			return 0
		

##########################################################################
#************************************************************************#
##########################################################################

class BBV_gradient_Richard_test2(BBV_gradient_Richard_test):

	def simulate(self):
		# Purpose: Simulates movement 
		t = self.t
		dt = t[1] - t[0]
		
		flag = 0
		
		# initialize arrays
		sol = np.zeros((len(t),3))
		p = np.zeros((len(t)))
		GOAL = np.zeros((len(t),self.ncompass))
		Ds = [0]
		Messages = ['start']

		# checking
		if self.mode == 1:
			optims = np.zeros((len(t)))
			optim_count = np.zeros(self.ncompass)
		elif self.mode == 2:
			pass

		# initialize values
		phi1 = -np.pi/3
		sol[0,:] = np.array([58, 2, phi1])
		p[0] = 1/self.h(self.field(sol[0,0]), self.a,self.b)
		GOAL[0,:] = 0.175*np.cos(self.EPG_dirs-phi1)
		
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
			
			sL = self.field(xyLA[0])
			sR = self.field(xyRA[0])
			hL, hR = self.h(sL,self.a,self.b), self.h(sR,self.a,self.b)

			hAvg = (hL + hR)/2
			p[i+1] = p[i] + dt*(-p[i] + 1/(hAvg))/self.tp
			d = p[i]*hAvg - 1
			Ds.append(d)

			optim = self.EPG_dirs[np.argmax(GOAL[i,:])]
			dtheta = math.remainder(theta - optim, 2*math.pi)

			if flag == 0:
				vL = self.v0 
				vR = self.v0 
			else:
				vL = self.v0 + dtheta/4 # + dtheta/4 + gam[i]
				vR = self.v0 - dtheta/4 # - dtheta/4 - gam[i]

			fx = [0.5*(vL+vR)*np.cos(theta), 0.5*(vL+vR)*np.sin(theta), (vR-vL)/self.d]
			

			if (x<45 and flag==0):
				phi2 = -2*np.pi/3
				theta = phi2
				flag = 1
			
			if flag==0:
				Messages.append('init angle='+str(round(180/np.pi*phi1)))
			else:
				Messages.append('FLIPPED!! to angle='+str(round(180/np.pi*phi2)))


			sol[i+1,:] = np.array([x,y,theta]) + dt*np.array(fx)
			EPG = np.cos(self.EPG_dirs-theta)
			GOAL[i+1,:] = GOAL[i,:] + dt*(-self.invtau*GOAL[i,:] - d*EPG)
			#GOAL[i+1,:][GOAL[i+1,:]<0]=0


			if self.mode == 1:
				optims[i+1] = self.EPG_dirs[np.argmax(GOAL[i+1,:])]
				optim_count[np.argmax(GOAL[i+1,:])] +=1
			elif self.mode == 2:
				pass
		
		if self.mode == 1:
			return sol, optim, optim_count, GOAL, np.array(Ds), Messages
		elif self.mode == 2:
			return 0
		

##########################################################################
#************************************************************************#
##########################################################################

class BBV_gradient_Richard_test3(BBV_gradient_Richard_test):

	def simulate(self):
		# Purpose: Simulates movement 
		t = self.t
		dt = t[1] - t[0]
		
		flag = 0
		
		# initialize arrays
		sol = np.zeros((len(t),3))
		p = np.zeros((len(t)))
		GOAL = np.zeros((len(t),self.ncompass))
		Messages = ['start']

		# checking
		optims = np.zeros((len(t)))
		optim_count = np.zeros(self.ncompass)

		# initialize values
		phi1 = np.pi/6
		sol[0,:] = np.array([2, 2, phi1])
		p[0] = 1/self.h(self.field(sol[0,0]), self.a,self.b)
		GOAL[0,:] = 4*np.cos(self.EPG_dirs-phi1)
		d = -np.cos(phi1)
		Ds = [-np.cos(phi1)]

		gam = self.ou(self.tau_m, self.sig_m, self.t)

		for i in range(len(t)-1):
			x,y,theta = sol[i,:]

			# NO BOUNDS
			# if self.outbounds(xyLA, xyRA):
			# 	x,y,theta = self.reflect(x,y,theta, xyLA, xyRA)
			# 	xyLA = self.get_antL_loc(x,y,theta)
			# 	xyRA = self.get_antR_loc(x,y,theta)

			optim = self.EPG_dirs[np.argmax(GOAL[i,:])]
			dtheta = math.remainder(theta - optim, 2*math.pi)

			if flag == 0:
				vL = self.v0 + self.mode*dtheta/4
				vR = self.v0 - self.mode*dtheta/4
			else:
				vL = self.v0 + 1*self.mode*dtheta/4 #+ gam[i] 
				vR = self.v0 - 1*self.mode*dtheta/4 #- gam[i] 

			fx = [0.5*(vL+vR)*np.cos(theta), 0.5*(vL+vR)*np.sin(theta), (vR-vL)/self.d]

			if (x>15 and flag==0):
				phi1 = -np.pi/3
				theta = phi1
				flag = 1
			
			if flag==0:
				Messages.append('Initial angle='+str(round(180/np.pi*phi1)))
			else:
				Messages.append('FLIPPED!! to angle='+str(round(180/np.pi*phi1)))

			#d = -np.cos(theta)
			#Ds.append(d)
			d = d + 0.1*(-d -np.cos(theta))
			Ds.append(d)
			sol[i+1,:] = np.array([x,y,theta]) + dt*np.array(fx)
			EPG = np.cos(self.EPG_dirs-theta)
			GOAL[i+1,:] = GOAL[i,:] + dt*(-self.invtau*GOAL[i,:] - d*EPG)
			#GOAL[i+1,:][GOAL[i+1,:]<0]=0

			optims[i+1] = self.EPG_dirs[np.argmax(GOAL[i+1,:])]
			optim_count[np.argmax(GOAL[i+1,:])] +=1
			
	
		return sol, optim, optim_count, GOAL, np.array(Ds), Messages
