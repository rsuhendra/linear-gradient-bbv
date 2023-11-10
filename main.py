from BBV_twochoice import *
from BBV_gradient import *
from BBV_gradient2 import *
import matplotlib.pyplot as plt
import math 
from collections import Counter

inputVals = np.array([0.725, -0.56, 0.5044506914904836, 0.3887560820695657, 0.6485170151559537, 4*0.3907077496945611, 0.75, 0.067])

# test = BBV_gradient_Josh_simple(weights=inputVals)
# ans = test.simulate()

# figname = 'Simple_tau4'
# fig, ax = plt.subplots(2, figsize=(5,8))
# ax[0].plot(ans[:,0],ans[:,1])
# ax[1].plot(ans[:,0])
# fig.savefig(figname+'.png')

t0 = np.linspace(24.5, 38, num=6001)
x2 = np.linspace(0, 600, num=6001)


#np.linspace(0, 350, )



test = BBV_gradient_Richard_test(weights=inputVals, tau=4)
ans, opt, optcount, goals = test.simulate()

# figname = 'Richard_d'
# fig, ax = plt.subplots(2,2)
# ax[0][0].plot(ans[:,0],ans[:,1])
# ax[0][1].plot(ans[:,0])
# ax[1][0].plot(opt)
# ax[1][1].hist(opt, bins=16)
# fig.savefig(figname+'.png')

# N = test.ncompass
# theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
# radii = optcount
# width = 2*np.pi / N
# ax = plt.subplot(projection='polar')
# ax.bar(theta, radii, width=width, bottom=0.0, alpha=0.5)
# plt.savefig(figname+'_polar.png')
# plt.close()

# # test
# directions = [math.remainder(x, 2*math.pi) for x in ans[:,2]]
# radii, _ = np.histogram(directions, bins = np.linspace(0, 2 * np.pi, N+1))
# theta = np.linspace(-np.pi, np.pi, N, endpoint=False)
# width = 2*np.pi / N
# ax = plt.subplot(projection='polar')
# ax.bar(theta, radii, width=width, bottom=0.0, alpha=0.5)
# plt.savefig(figname+'ablate_polar.png')

