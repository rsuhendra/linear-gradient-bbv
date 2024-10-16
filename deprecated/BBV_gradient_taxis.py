from BBV_gradient import *

def angle_diff(theta1, theta2):
	diff = np.abs(theta1 - theta2)%(2*np.pi)
	# diff = diff if diff < np.pi else 2*np.pi - diff
	diff = diff + (diff>np.pi)*(2*np.pi - 2*diff)
	return diff

class BBV_gradient_taxis_base(BBV_gradient_base):

	def __init__(self, weights):
		BBV_gradient_base.__init__(self)
		self.v0 = 2

		self.T = 600
		self.freq = 30
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
	
	def outbounds_circle(self, xyLA, xyRA):
		center = np.array([17.5, 0])
		LAout = np.linalg.norm(xyLA - center) > 17.5
		RAout = np.linalg.norm(xyRA - center) > 17.5
		return LAout or RAout

class BBV_gradient_taxis_angles(BBV_gradient_taxis_base):

	def __init__(self, weights):
		BBV_gradient_taxis_base.__init__(self, weights)

	def simulate(self):
		# Purpose: Simulates movement 
		t = self.t
		dt = t[1] - t[0]

		# initialize arrays
		sol = np.zeros((len(t),3))
		p = np.zeros((len(t)))

		# initialize values
		sol[0,:] = np.array([17.5, 0, 2*np.pi*np.random.rand()])
		p[0] = 1/self.h(self.field(sol[0,0]), self.a,self.b)
		
		epsL = self.ou(self.tau_s, self.sig_s, self.t)
		epsR = self.ou(self.tau_s, self.sig_s, self.t)
		gam = self.ou(self.tau_m, self.sig_m, self.t)

		endPoint = len(t)

		for i in range(len(t)-1):
			x,y,theta = sol[i,:]
			xyLA = self.get_antL_loc(x,y,theta)
			xyRA = self.get_antR_loc(x,y,theta)

			# Reflect if either antenna hitting wall
			if self.outbounds_circle(xyLA, xyRA):
				endPoint = i
				break
			
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

		return sol[:endPoint,:]

	
class BBV_gradient_taxis_speed(BBV_gradient_taxis_base):

	def __init__(self, weights):
		BBV_gradient_taxis_base.__init__(self, weights)
		
	def simulate(self):
		# Purpose: Simulates movement 
		t = self.t
		dt = t[1] - t[0]

		# initialize arrays
		sol = np.zeros((len(t),3))
		p = np.zeros((len(t)))

		# initialize values
		sol[0,:] = np.array([17.5, 0, 2*np.pi*np.random.rand()])
		p[0] = 1/self.h(self.field(sol[0,0]), self.a,self.b)
		
		epsL = self.ou(self.tau_s, self.sig_s, self.t)
		epsR = self.ou(self.tau_s, self.sig_s, self.t)
		gam = self.ou(self.tau_m, self.sig_m, self.t)

		endPoint = len(t)

		for i in range(len(t)-1):
			x,y,theta = sol[i,:]
			xyLA = self.get_antL_loc(x,y,theta)
			xyRA = self.get_antR_loc(x,y,theta)

			# Reflect if either antenna hitting wall
			if self.outbounds_circle(xyLA, xyRA):
				endPoint = i
				break
			
			sL = self.field(xyLA[0]) + epsL[i]
			sR = self.field(xyRA[0]) + epsR[i]

			hL, hR = self.h(sL,self.a,self.b), self.h(sR,self.a,self.b)

			hAvg = (hL + hR)/2
			p[i+1] = p[i] + dt*(-p[i] + 1/(hAvg))/self.tp
			d = p[i]*hAvg - 1
			tax = self.h(-d, a=20, b=0)
			
			vL = self.wI * hL + self.wC * hR + self.v0 + gam[i] + 4*tax
			vR = self.wC * hL + self.wI * hR + self.v0 - gam[i] + 4*tax

			# vL = self.v0 + gam[i] + 4*tax
			# vR = self.v0 - gam[i] + 4*tax

			fx = [0.5*(vL+vR)*np.cos(theta), 0.5*(vL+vR)*np.sin(theta), (vR-vL)/self.d]
			sol[i+1,:] = np.array([x,y,theta]) + dt*np.array(fx)

		return sol[:endPoint,:]


class BBV_gradient_taxis_both(BBV_gradient_taxis_base):

	# rewrite weights to simulate
	def __init__(self, weights, p1, p2):
		BBV_gradient_taxis_base.__init__(self, weights)
		self.p1 = p1
		self.p2 = p2
	def simulate(self):
		# Purpose: Simulates movement 
		t = self.t
		dt = t[1] - t[0]

		# initialize arrays
		sol = np.zeros((len(t),3))
		p = np.zeros((len(t)))
		taxes = []

		# initialize values
		sol[0,:] = np.array([17.5, 0, 2*np.pi*np.random.rand()])
		# p[0] = 1/self.h(self.field(sol[0,0]), self.a,self.b)
	

		epsL = self.ou(self.tau_s, self.sig_s, self.t)
		epsR = self.ou(self.tau_s, self.sig_s, self.t)
		gam = self.ou(self.tau_m, self.sig_m, self.t)

		endPoint = len(t)

		p1, p2 = self.p1, self.p2

		for i in range(len(t)-1):
			x,y,theta = sol[i,:]
			xyLA = self.get_antL_loc(x,y,theta)
			xyRA = self.get_antR_loc(x,y,theta)

			# Reflect if either antenna hitting wall
			if self.outbounds_circle(xyLA, xyRA):
				endPoint = i
				break
			
			sL = self.field(xyLA[0]) #+ epsL[i]
			sR = self.field(xyRA[0]) #+ epsR[i]

			hL, hR = self.h(sL,self.a,self.b), self.h(sR,self.a,self.b)
			hAvg = (hL + hR)/2

			if i == 0:
				p[0] = 1/hAvg

			p[i+1] = p[i] + dt*(-p[i] + 1/(hAvg))/self.tp
			d = p[i]*hAvg - 1
			tax = self.h(d, a=50, b=0)
			# tax = angle_diff(0, theta)/np.pi
			taxes.append(tax)

			# vL = self.wI * hL + self.wC * hR + self.v0 + p1*tax*gam[i] + p2*tax
			# vR = self.wC * hL + self.wI * hR + self.v0 - tax*gam[i] + 4*tax

			if p1 == 0:
				vL = self.v0 + gam[i] + p2*(1-tax)
				vR = self.v0 - gam[i] + p2*(1-tax)
			else:
				vL = self.v0 + (0.25+0.75*tax)*gam[i] + p2*(1-tax)
				vR = self.v0 - (0.25+0.75*tax)*gam[i] + p2*(1-tax)

			fx = [0.5*(vL+vR)*np.cos(theta), 0.5*(vL+vR)*np.sin(theta), (vR-vL)/self.d]
			fx = np.array(fx)/10 # account for cm to mm conversion
			sol[i+1,:] = np.array([x,y,theta]) + dt*np.array(fx)

		return sol[:endPoint,:]