from BBV_gradient import *
from BBV_compass import *
import matplotlib.pyplot as plt
import math 
import deap
from collections import Counter
from functions_plotting import *

# name = 'Perfect_p1=0.75'
# output_file = 'perfect_p1=0.75.output'

output_file = 'outputs/test7.output'
name = 'Chemotaxis model'

turn_thresh = 2*np.pi/3

percent_reached_plot(plot_name = name, output_file=output_file)
sample_plot(plot_name = name, output_file=output_file, ht = turn_thresh)
polar_turns(ht=turn_thresh, plot_name = name, output_file=output_file)
num_turns_direction(ht=turn_thresh, plot_name = name, output_file=output_file)
average_velocity(plot_name = name, output_file=output_file)
turn_distribution(ht=turn_thresh, plot_name = name, output_file=output_file)


# figname = 'Simple_tau4'
# fig, ax = plt.subplots(2, figsize=(5,8))
# ax[0].plot(ans[:,0],ans[:,1])
# ax[1].plot(ans[:,0])
# fig.savefig(figname+'.png')

# t0 = np.linspace(24.5, 38, num=6001)
# x2 = np.linspace(0, 600, num=6001)



# test1 = BBV_gradient_Richard_taxis(weights=inputVals)
# test2 = BBV_gradient_Josh_simple(weights=inputVals)
# test3 = BBV_gradient_Richard_full(weights=inputVals, tau=4)

# bbv.sample_plot()
# bbv.percent_reached_plot(numSimulations=20)
# bbv.average_velocity(numSimulations=20)

# tests = [test1, test2, test3]
# for test in tests:
# 	test.v0 = 2
# 	test.sample_plot()
	# test.percent_reached_plot()

# test3.average_velocity(numSimulations=200)



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

