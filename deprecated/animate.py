from BBV_twochoice import *
from BBV_gradient import *
from BBV_gradient_taxis import *
import matplotlib.pyplot as plt
import math 
from collections import Counter
import matplotlib.animation as animation

inputVals = np.array([0.725, -0.56, 0.5044506914904836, 0.3887560820695657, 0.6485170151559537, 4*0.3907077496945611, 0.75, 0.067])
lineprojd = 0.5

test = BBV_gradient_Richard_taxis(weights=inputVals)

ans = test.simulate()

fig = plt.figure()
ax1 = plt.subplot(111)

# plotting position + angle
scat = ax1.scatter(ans[0,0], ans[1,0])
ax1.set(xlim=[0, test.stageW], ylim=[0, test.stageH])
ax1.set_aspect('equal')
ax1.legend()

xdir, ydir = test.line_projection(ans[0,0], ans[0,1], ans[0,2], d=lineprojd)
plot_dir, = ax1.plot(xdir, ydir)
ax1.set_title('position + angle(line poking out)')

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

	return scat

ani = animation.FuncAnimation(fig=fig, func=update, frames=1800, interval=30)
writervideo = animation.FFMpegWriter(fps=60) 
#ani.save('test.mp4', writer=writervideo) 
plt.show()
