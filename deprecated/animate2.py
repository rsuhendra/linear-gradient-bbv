from BBV_twochoice import *
from BBV_gradient import *
from BBV_gradient_taxis import *
import matplotlib.pyplot as plt
import math 
from collections import Counter
import matplotlib.animation as animation

inputVals = np.array([0.725, -0.56, 0.5044506914904836, 0.3887560820695657, 0.6485170151559537, 4*0.3907077496945611, 0.75, 0.067])
lineprojd = 5

test = BBV_gradient_Richard_test2(weights=inputVals, tau=4)
test.stageW = 60
test.stageH = 60
ans, opt, optcount, goals, ds, msgs = test.simulate()

thetas_toplot = np.linspace(0, 2*np.pi, test.ncompass+1)

fig = plt.figure()
ax1 = plt.subplot(221)
ax2 = plt.subplot(222, projection='polar')
ax3 = plt.subplot(223)
ax4 = plt.subplot(224, projection='polar')

# plotting position + angle
scat = ax1.scatter(ans[0,0], ans[1,0])
ax1.set(xlim=[0, test.stageW], ylim=[0, test.stageH])
ax1.set_aspect('equal')
ax1.legend()

xdir, ydir = test.line_projection(ans[0,0], ans[0,1], ans[0,2], d=lineprojd)
plot_dir, = ax1.plot(xdir, ydir)
ax1.set_title('position + angle(line poking out)')

# plotting goal
goals_plot = np.zeros((goals.shape[0], goals.shape[1]+1))
goals_plot[:,:-1] = goals
goals_plot[:,-1] = goals[:,0]

polplot, = ax2.plot(thetas_toplot, goals_plot[0,:])
ax2.set(ylim=[np.min(goals), np.max(goals)])
max_angle_k = np.argmax(goals_plot[0,:])
polscatter = ax2.scatter(thetas_toplot[max_angle_k],goals_plot[0,max_angle_k], color='red')
ax2.set_title('Goal direction')

# plotting d 
plot3, = ax3.plot(test.t, ds)
plot3, = ax3.plot(test.t, np.zeros(test.t.shape), 'k--')
scat3 = ax3.scatter(test.t[0], ds[0], color='red')
ax3.set_title('"derivative" d')

# plotting EPG
polplot2, = ax4.plot(thetas_toplot, np.cos(thetas_toplot-ans[0,2]))
ax4.set(ylim=[-1, 1.1])
max_angle_k = np.argmax(np.cos(thetas_toplot-ans[0,2]))
polscatter2 = ax4.scatter(thetas_toplot[max_angle_k],np.cos(thetas_toplot-ans[0,2])[max_angle_k], color='red')
ax4.set_title('EPG direction')

fig.suptitle('test')
fig.tight_layout()

def update(frame):
	# for each frame, update the data stored on each artist.
	x = ans[frame,0]
	y = ans[frame,1]
	# update the scatter plot:
	data = np.stack([x, y]).T
	scat.set_offsets(data)

	# update line direction
	xdir, ydir = test.line_projection(ans[frame,0], ans[frame,1], ans[frame,2], d=lineprojd)
	plot_dir.set_xdata(xdir)
	plot_dir.set_ydata(ydir)

	# update polar plot
	polplot.set_ydata(goals_plot[frame,:])
	max_angle_k = np.argmax(goals_plot[frame,:])
	polscatter.set_offsets([thetas_toplot[max_angle_k],goals_plot[frame,max_angle_k]])

	# update d plot
	scat3.set_offsets([test.t[frame], ds[frame]])

	# update EPG plot
	polplot2.set_ydata(np.cos(thetas_toplot-ans[frame,2]))
	max_angle_k = np.argmax(np.cos(thetas_toplot-ans[frame,2]))
	polscatter2.set_offsets([thetas_toplot[max_angle_k],np.cos(thetas_toplot-ans[frame,2])[max_angle_k]])

	fig.suptitle(msgs[frame])

	return scat

ani = animation.FuncAnimation(fig=fig, func=update, frames=600, interval=30)
writervideo = animation.FFMpegWriter(fps=60) 
ani.save('fakebounds_yes_correction_no_noise_3.mp4', writer=writervideo) 
plt.show()






# fig, ax = plt.subplots()

# scat = ax.scatter(ans[0,0], ans[1,0])
# ax.set(xlim=[0, 35], ylim=[0, 20])
# ax.legend()

# def update(frame):
#     # for each frame, update the data stored on each artist.
#     x = ans[:frame,0]
#     y = ans[:frame,1]
#     # update the scatter plot:
#     data = np.stack([x, y]).T
#     scat.set_offsets(data)

#     return (scat)

# ani = animation.FuncAnimation(fig=fig, func=update, frames=1800, interval=30)
# plt.show()