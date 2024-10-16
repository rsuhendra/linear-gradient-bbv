from BBV_twochoice import *
from BBV_compass import *
import matplotlib.pyplot as plt
import seaborn as sns

# inputVals = np.array([0.725, -0.56, 0.5, 0.39, 0.65, 0.45, 0.75, 0.067])

inputVals = np.array([0.725, -0.56, 0.5, 0.39, 0.65, 0.05, 0.75, 0.067])

bbv = BBV_gradient_taxis_point(weights=inputVals, p1=0.75, p2=0, brait=True, mode_motor='ou', mode_taxis='derivative')

bbv.generate_data(output_name='test7.output', num_Simulations=100)

# bbv = BBV_gradient_compass_point(weights=inputVals, p1=0, p2=0, p3=0, mode_motor='ou', mode_taxis='perfect')


# sol, vels, GOAL = bbv.simulate(testing=True)
# levels = np.max(GOAL, axis=1)
# direcs = []
# for i in range(len(GOAL)):
# 	direcs.append(calc_direction(GOAL[i,:], bbv.EPG_dirs))

# # plt.plot(cast_to(sol[:,2]))
# # plt.plot(cast_to(np.array(direcs)))
# plt.plot(levels)
# plt.show()
# plt.close()




# sol, taxes = bbv.simulate()
# # plt.plot(sol[:,0])
# # plt.plot(20*np.array(taxes))

# sns.regplot(x=taxes, y=angle_diff(0,sol[:,2]), order=2)

# plt.show()

