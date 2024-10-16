
class BBV_gradient_Josh_F(BBV_gradient_base):

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

		self.ncompass = 16
		self.thetas = np.linspace(0,2*np.pi,self.ncompass, endpoint=False)
		self.thetaVs = np.vstack((np.cos(self.thetas),np.sin(self.thetas))).T
		self.capacity = 15

	def simulate(self):
		# Purpose: Simulates movement 
		t = self.t
		dt = t[1] - t[0]
		sol = np.zeros((len(t),3))
		sol[0,:] = np.array([3.5, 10, 2*np.pi*np.random.rand()])

		# Memory for previous position and temp
		prevPos = MemoryBuffer(self.capacity)
		prevTemp = MemoryBuffer(self.capacity)
		prevPos.push(sol[0,0:2])
		prevTemp.push(self.field(sol[0,0]))
		optim = np.zeros((len(t)))
		optim_count = np.zeros(self.ncompass)
		
		# initialize OU processes
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

			# calculate direction of max F-stat
			fstats = []
			for k in range(self.ncompass):
				xvals = np.dot(prevPos.arrayize(),self.thetaVs[k])
				data = pd.DataFrame(np.vstack((xvals,prevTemp.arrayize())).T,columns=['xproj','Tdat'])
				if data.shape[0]>1:
					#calculate least squares fit, corresponding f-stat
					## calculate f-statistic. 
					lm = ols('Tdat ~ xproj',data=data).fit()
					fstat = (lm.fvalue if lm.params['xproj']<0 else 0)
					fstats.append(fstat)
				else:
					fstats.append(np.random.randn(1))
			optim[i] = self.thetas[np.argmax(fstats)]
			optim_count[np.argmax(fstats)] +=1
			# update memory
			prevTemp.push((sL+sR)/2)
			prevPos.push(np.array([x,y]))

			vL = self.wI * hL + self.wC * hR + self.v0 + gam[i]
			vR = self.wC * hL + self.wI * hR + self.v0 - gam[i] 

			fx = [0.5*(vL+vR)*np.cos(theta), 0.5*(vL+vR)*np.sin(theta), (vR-vL)/self.d]
			sol[i+1,:] = np.array([x,y,theta]) + dt*np.array(fx)

		return sol, optim, optim_count
	