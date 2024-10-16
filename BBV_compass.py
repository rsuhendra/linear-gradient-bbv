from BBV_gradient import *


class BBV_gradient_compass_point(BBV_gradient_taxis):

	def __init__(self, weights, brait, mode_taxis = 'perfect', mode_motor = 'ou' ):
		BBV_gradient_taxis.__init__(self, weights, mode_taxis, mode_motor)

		self.v0 = 5

		self.T = 600
		self.freq = 30
		self.dt = 1/self.freq
		self.t = np.linspace(0, self.T, self.T * self.freq, endpoint=False)

		self.brait = brait

		self.tau_goal = 1
		self.ncompass = 12
		self.EPG_dirs = np.linspace(0, 2*np.pi, self.ncompass, endpoint=False)

		self.amp = 5

		# Local Cosine
		# self.g = 1.75
		# alpha, beta, D = 2.0, 1.0, 0.3
		# # alpha, beta, D = 2.0, 0.5, 1.0
		# Wrow = D*np.cos(np.arange(self.ncompass) * 2 * np.pi / self.ncompass)
		# Wrow = ReLU(Wrow) - beta
		# Wrow[0] += alpha

		# Cosine
		self.g = 1.75
		Wrow = np.cos(np.arange(self.ncompass) * 2 * np.pi / self.ncompass)

		# Construct Wmat
		self.Wmat = np.array([np.roll(Wrow, n) for n in range(self.ncompass)])

	def ringrhs(self, v, Wmat, tau, g, N, input):
		recurr = (2 * g / N) * np.dot(Wmat, v)
		nonlin = ReLU(recurr + input)
		dydt = (-v + nonlin) / tau
		return dydt
	
	def epg_shift(self, theta, shift = 0, add = 0):
		return add + np.cos(theta - self.EPG_dirs + shift)

	def simulate(self, testing = False):
		# Purpose: Simulates movement 
		t = self.t
		dt = t[1] - t[0]

		# initialize arrays
		sol = np.zeros((len(t),3))
		vels = np.zeros((len(t),3))
		p = np.zeros((len(t)))
		GOAL = np.zeros((len(t),self.ncompass))
		# taxes = []

		max_lambda = 0.5
		time_counter = 0
		point_time = np.random.default_rng().exponential(scale=max_lambda)
		triang_pause = False

		# initialize values
		sol[0,:] = np.array([35, 100, 2*np.pi*np.random.rand()])
		p[0] = 1/self.h(self.field(sol[0,0]), self.a,self.b)
		
		epsL = self.ou(self.tau_s, self.sig_s, self.t)
		epsR = self.ou(self.tau_s, self.sig_s, self.t)

		if self.mode_motor == 'ou':
			gam = self.ou(self.tau_m, self.sig_m, self.t)
		elif self.mode_motor == 'signed_ou':
			gam = self.ou(self.tau_m, self.sig_m, self.t, mode = 'signed')

		for i in range(len(t)-1):
			x,y,theta = sol[i,:]
			xyLA = self.get_antL_loc(x,y,theta)
			xyRA = self.get_antR_loc(x,y,theta)

			# If past a certain point stop sim
			if x > 0.85*self.stageW:
				if testing == True:
					return sol[:i,:], vels[:i,:], GOAL[:i,:]
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

			goal_norm = GOAL[i,:] - np.mean(GOAL[i,:])
			m_A = 15

			pfl3_l = ELU(m_A * goal_norm + self.epg_shift(theta, shift = 90*(180/np.pi), add = 0))
			pfl3_r = ELU(m_A * goal_norm + self.epg_shift(theta, shift = -90.0*(180/np.pi), add = 0))
			pfl2 = ELU(m_A * goal_norm + self.epg_shift(theta, shift = np.pi, add = 0))

			# Find a better way to do this
			if time_counter > point_time:
				
				# Check PFL2 activity if turn is "surpressed"
				sgn = self.h(np.mean(pfl2), a=10, b=1)
				if np.random.rand()>sgn:
					time_counter = 0
					triang_pause = False
					point_time = np.random.default_rng().exponential(scale=max_lambda)
				
				# If turn "happens"
				else:
					time_counter = 0
					triang_pause = True

					# Check PFL3 activity for sign of turn
					pfl3_diff = self.h(np.mean(pfl3_l) - np.mean(pfl3_r),10,0)
					sgn = 2*(np.random.rand() < pfl3_diff)-1
     
					# sgn = 2*np.random.randint(2) - 1
					# sgn = (1+np.sin(theta - optim))/2
					# print(optim, theta%(2*np.pi), np.sin(theta - optim))

					# Simulate turn 
					mu, sigma = 2, 0.5 # mean and standard deviation
					s = 0.25 + np.random.lognormal(mu, sigma)	
					# sawtooth_time = np.linspace(0, 1, 1+int(s))
					sawtooth_time = np.arange(start=0, stop=1, step=1/s)
					sawtooth_time = (1-sawtooth_time[-1])/2 + sawtooth_time
					triangle = sgn*(np.pi/2)*(1+signal.sawtooth(2 * np.pi * sawtooth_time,  width = 0.5))

					counter = 0

			vr_add = 0
			if triang_pause == True:
				vr_add = triangle[counter]
				counter +=1
    
				if counter >= len(sawtooth_time):
					triang_pause = False
					point_time = np.random.default_rng().exponential(scale=max_lambda)
			else:
				time_counter += dt

			vL = self.v0 + vr_add #- tax_mult*gam[i] + p2*(1-tax)
			vR = self.v0 - vr_add #+ tax_mult*gam[i] + p2*(1-tax) 

			if self.brait == True:
				vL += self.wI * hL + self.wC * hR
				vR += self.wC * hL + self.wI * hR

			fx = [0.5*(vL+vR)*np.cos(theta), 0.5*(vL+vR)*np.sin(theta), (vR-vL)/self.d]
			fx = np.array(fx)
			if i == 0:
				vels[0,:] = fx
			vels[i+1,:] = fx
			sol[i+1,:] = np.array([x,y,theta]) + dt*np.array(fx)

			# EPG = 1 + np.cos(self.EPG_dirs-theta)
			# a, b = 1, 0.1
			# GOAL[i+1,:] = GOAL[i,:] + dt*(-GOAL[i,:] + a*(np.roll(GOAL[i,:], 1) + np.roll(GOAL[i,:], -1)) - b*np.sum(GOAL[i,:]) - d*EPG)/self.tau_goal

			input = self.amp * self.epg_shift(theta) * ReLU(-d)
			input += self.amp * self.epg_shift(theta, shift=np.pi) * ReLU(d)

			# input = self.epg_shift(theta, add=0) + d
   
			GOAL[i+1,:] = GOAL[i,:] + dt * self.ringrhs(GOAL[i,:], self.Wmat, self.tau_goal, self.g, self.ncompass, input)

		if testing == True:
			return sol, vels, GOAL
		else:
			return sol, vels

##########################################################################
#************************************************************************#
##########################################################################


class BBV_gradient_compass(BBV_gradient_taxis):
    

	def __init__(self, weights, p1, p2, p3, mode_taxis = 'perfect', mode_motor = 'ou' ):
		BBV_gradient_taxis.__init__(self, weights, p1, p2, mode_taxis, mode_motor)

		self.invtau = 0.2
		self.p3 = p3
		self.ncompass = 16
		self.EPG_dirs = np.linspace(0, 2*np.pi, self.ncompass, endpoint=False)

	def simulate(self):
		# Purpose: Simulates movement 
		t = self.t
		dt = t[1] - t[0]

		# initialize arrays
		sol = np.zeros((len(t),3))
		vels = np.zeros((len(t),3))
		p = np.zeros((len(t)))
		GOAL = np.zeros((len(t),self.ncompass))

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

		p1, p2, p3 = self.p1, self.p2, self.p3

		for i in range(len(t)-1):
			x,y,theta = sol[i,:]
			xyLA = self.get_antL_loc(x,y,theta)
			xyRA = self.get_antR_loc(x,y,theta)

			# If past a certain point stop sim
			if x > 0.85*self.stageW:
				return sol[:i,:], vels[:i,:], GOAL[:i,:]

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

			optim = self.EPG_dirs[np.argmax(GOAL[i,:])]
			dtheta = math.remainder(theta - optim, 2*math.pi)

			tax_mult = (1-p1) + p1*tax   # p1==0 equates to regular
			vL = self.v0 + tax_mult*gam[i] + p2*(1-tax) + p3*dtheta
			vR = self.v0 - tax_mult*gam[i] + p2*(1-tax) - p3*dtheta

			fx = [0.5*(vL+vR)*np.cos(theta), 0.5*(vL+vR)*np.sin(theta), (vR-vL)/self.d]
			fx = np.array(fx)
			if i == 0:
				vels[0,:] = fx
			vels[i+1,:] = fx
			sol[i+1,:] = np.array([x,y,theta]) + dt*np.array(fx)
			EPG = np.cos(self.EPG_dirs-theta)
			GOAL[i+1,:] = GOAL[i,:] + dt*(-self.invtau*GOAL[i,:] - d*EPG)

		return sol, vels, GOAL





### DEPRECATED


class BBV_gradient_compass_point_OLD(BBV_gradient_taxis):

	def __init__(self, weights, p1, p2, brait, mode_taxis = 'perfect', mode_motor = 'ou' ):
		BBV_gradient_taxis.__init__(self, weights, p1, p2, mode_taxis, mode_motor)

		self.v0 = 5

		self.T = 600
		self.freq = 30
		self.dt = 1/self.freq
		self.t = np.linspace(0, self.T, self.T * self.freq, endpoint=False)

		self.brait = brait

		self.tau_goal = 1
		self.ncompass = 12
		self.EPG_dirs = np.linspace(0, 2*np.pi, self.ncompass, endpoint=False)

		self.amp = 5

		# Local Cosine
		# self.g = 1.75
		# alpha, beta, D = 2.0, 1.0, 0.3
		# # alpha, beta, D = 2.0, 0.5, 1.0
		# Wrow = D*np.cos(np.arange(self.ncompass) * 2 * np.pi / self.ncompass)
		# Wrow = ReLU(Wrow) - beta
		# Wrow[0] += alpha

		# Cosine
		self.g = 1.75
		Wrow = np.cos(np.arange(self.ncompass) * 2 * np.pi / self.ncompass)

		# Construct Wmat
		self.Wmat = np.array([np.roll(Wrow, n) for n in range(self.ncompass)])

	def ringrhs(self, v, Wmat, tau, g, N, input):
		recurr = (2 * g / N) * np.dot(Wmat, v)
		nonlin = ReLU(recurr + input)
		dydt = (-v + nonlin) / tau
		return dydt
	
	def epg_shift(self, theta, shift = 0, add = 0):
		return add + np.cos(theta - self.EPG_dirs + shift)

	def simulate(self, testing = False):
		# Purpose: Simulates movement 
		t = self.t
		dt = t[1] - t[0]

		# initialize arrays
		sol = np.zeros((len(t),3))
		vels = np.zeros((len(t),3))
		p = np.zeros((len(t)))
		GOAL = np.zeros((len(t),self.ncompass))
		# taxes = []

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
					return sol[:i,:], vels[:i,:], GOAL[:i,:]
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

			# if self.mode_taxis == 'derivative':
			# 	tax = self.h(d, a=200, b=0)
			# 	# print(200*d, tax)
			# elif self.mode_taxis == 'perfect':
			# 	tax = angle_diff(0, theta)/np.pi
	
			# taxes.append(tax)
			# optim = self.EPG_dirs[np.argmax(GOAL[i,:])]
			# optim = calc_direction(GOAL[i,:], self.EPG_dirs)

			goal_norm = GOAL[i,:] - np.mean(GOAL[i,:])
			m_A = 15

			pfl3_l = ELU(m_A * goal_norm + self.epg_shift(theta, shift = 90*(180/np.pi), add = 0))
			pfl3_r = ELU(m_A * goal_norm + self.epg_shift(theta, shift = -90.0*(180/np.pi), add = 0))
			pfl2 = ELU(m_A * goal_norm + self.epg_shift(theta, shift = np.pi, add = 0))

			# tax_mult = (1-p1) + p1*tax   # p1==0 equates to regular

			# Find a better way to do this
			if time_counter > point_time:
				time_counter = 0
				pause = True
				point_time = np.random.default_rng().exponential(scale=max_lambda)
				mu, sigma = 2, 0.5 # mean and standard deviation
				s = np.random.lognormal(mu, sigma)	
				test = np.linspace(0, 1, 1+int(s))
				# sgn = 2*np.random.randint(2) - 1
				# sgn = (1+np.sin(theta - optim))/2
				sgn = self.h(np.mean(pfl3_l) - np.mean(pfl3_r),10,0)
				# print(sgn)
				sgn = 2*(np.random.rand() < sgn)-1
				# print(optim, theta%(2*np.pi), np.sin(theta - optim))
				triangle = sgn*(np.pi/2)*(1+signal.sawtooth(2 * np.pi * test,  width = 0.5))

				sgn = self.h(np.mean(pfl2), a=10, b=1)
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

			# EPG = 1 + np.cos(self.EPG_dirs-theta)
			# a, b = 1, 0.1
			# GOAL[i+1,:] = GOAL[i,:] + dt*(-GOAL[i,:] + a*(np.roll(GOAL[i,:], 1) + np.roll(GOAL[i,:], -1)) - b*np.sum(GOAL[i,:]) - d*EPG)/self.tau_goal

			# input = self.amp * self.epg_shift(theta) * ReLU(-d)
			# input += self.amp * self.epg_shift(theta, shift=np.pi) * ReLU(d)

			input = self.epg_shift(theta, add=0) + d
   
			GOAL[i+1,:] = GOAL[i,:] + dt * self.ringrhs(GOAL[i,:], self.Wmat, self.tau_goal, self.g, self.ncompass, input)

		if testing == True:
			return sol, vels, GOAL
		else:
			return sol, vels

##########################################################################
#************************************************************************#
##########################################################################
