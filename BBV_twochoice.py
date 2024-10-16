from BBV import *
import pickle
from scipy.interpolate import griddata

class BBV_twochoice_base(BBV_base):

	def __init__(self, temp):
		BBV_base.__init__(self)

		self.temp = temp
		self.wallRadius = 19
	
	def outbounds(self, xyLA, xyRA):
		# Purpose: Check if antenna in bounds
		return np.linalg.norm(xyLA)>self.wallRadius or np.linalg.norm(xyRA)>self.wallRadius

	def reflect(self, x,y,theta):
		# Purpose: reflects the animal when it approaches the boundary
		# for circular two-choice arena

		# find intersection of body axis vector and circle
		vx,vy = np.cos(theta),np.sin(theta)
		a = (vx**2+vy**2)
		b = 2*(x*vx+y*vy) 
		c = x**2 + y**2 - self.wallRadius**2.
		if b**2 - 4*a*c <0:
			return theta+np.pi
		tArray = [(-b +np.sqrt(b**2 -4.*a*c))/(2.*a),(-b -np.sqrt(b**2 -4.*a*c))/(2.*a)]

		i1 = np.argmin(np.abs(tArray))
		pos = np.array([x,y]) + tArray[i1]*np.array([vx,vy])
		
		## perform reflection
		ang2 = np.arctan2(pos[1],pos[0])
		ang1 = np.arctan2(vy,vx)
		l1  = np.argmin(np.abs([ang1-ang2,ang1-ang2-2*np.pi,ang1-ang2+2*np.pi]))
		l0 = [ang1-ang2,ang1-ang2-2*np.pi,ang1-ang2+2*np.pi]
		l0 = l0[l1]

		thetaNew = theta + 2* np.sign(l0)*(np.pi/2.-np.abs(l0))

		return thetaNew
	
	def field(self, x,y):
		# Purpose: Returns arena temp at given position
		h1 = (x*y>0)*2 - 1
		d1 = np.min(np.abs(np.array([x,y])), axis=0)

		# fields is a global variable, stores the arena temp information
		tm = fields[self.temp]
		temp = griddata(fields['xv'],tm,d1*h1)

		return (temp-25.)/10.
	
	def simulate(self):
		# Purpose: Simulates movement (Placeholder function)
		return 0
	
##########################################################################
#************************************************************************#
##########################################################################

class BBV_twochoice_Josh(BBV_twochoice_base):

	def __init__(self, temp, weights):
		BBV_twochoice_base.__init__(self, temp)

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
		sol[0,:] = np.array([8, (np.random.randint(2)*2-1)*8, 2*np.pi*np.random.rand()])

		epsL = self.ou(self.tau_s, self.sig_s, self.t)
		epsR = self.ou(self.tau_s, self.sig_s, self.t)
		gam = self.ou(self.tau_m, self.sig_m, self.t)

		for i in range(len(t)-1):
			x,y,theta = sol[i,:]
			xyLA = self.get_antL_loc(x,y,theta)
			xyRA = self.get_antR_loc(x,y,theta)

			# Reflect if either antenna hitting wall
			if self.outbounds(xyLA, xyRA):
				theta = self.reflect(x,y,theta)
				xyLA = self.get_antL_loc(x,y,theta)
				xyRA = self.get_antR_loc(x,y,theta)

			sL = self.field(xyLA[0],xyLA[1]) + epsL[i]
			sR = self.field(xyRA[0],xyRA[1]) + epsR[i]
			hL, hR = self.h(sL,self.a,self.b), self.h(sR,self.a,self.b)

			vL = self.wI * hL + self.wC * hR + self.v0 + gam[i]
			vR = self.wC * hL + self.wI * hR + self.v0 - gam[i] 

			fx = [0.5*(vL+vR)*np.cos(theta), 0.5*(vL+vR)*np.sin(theta), (vR-vL)/self.d]
			sol[i+1,:] = np.array([x,y,theta]) + dt*np.array(fx)

		return sol

##########################################################################
#************************************************************************#
##########################################################################

# Need to initialize the file

(yvals30,levels30,yvals35,levels35,yvals40,levels40,x2,y2,
     ti,ti35,ti0) = pickle.load(open("contour_info30_40.pkl", "rb"), encoding="bytes")

fields = dict()
fields['xv'] = np.hstack((np.linspace(-21,-9,endpoint=False), x2,np.linspace(9, 21)))
fields[30] = np.hstack((np.ones(50)*25, ti[0],np.ones(50)*np.max(ti)))
fields[35] = np.hstack((np.ones(50)*25, ti35[0],np.ones(50)*np.max(ti35)))
fields[40] = np.hstack((np.ones(50)*25, ti0[0],np.ones(50)*np.max(ti0)))