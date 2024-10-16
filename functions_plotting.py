from utils import *
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import joypy
from scipy.signal import find_peaks
from matplotlib.cm import get_cmap
import matplotlib as mpl

C = np.loadtxt('cmap.txt',dtype='int').astype('float')
cm1 = C/255.0
cm1 = mpl.colors.ListedColormap(cm1)
(x2,t0) = pickle.load(open("contour_datBig_gradient.pkl","rb"),encoding='latin1')

def find_turn_indices(angVels, ht=np.pi/3, eps=0):

	turn_idxs = []
	peaks = []

	exceed_indices = np.where(angVels > ht)[0]
	result_segments = indices_grouped_by_condition(angVels, lambda x: x > eps)
	for seg in result_segments:
		if len(seg)<=2:
			continue
		turn = (seg[0], seg[-1])
		if is_number_in_interval(exceed_indices, turn):
			turn_idxs.append(turn)
			peaks.append(int((turn[0]+turn[1])/2))

	exceed_indices = np.where(angVels < -ht)[0]
	result_segments = indices_grouped_by_condition(angVels, lambda x: x < -eps)
	for seg in result_segments:
		if len(seg)<=2:
			continue
		turn = (seg[0], seg[-1])
		if is_number_in_interval(exceed_indices, turn):
			turn_idxs.append(turn)
			peaks.append(int((turn[0]+turn[1])/2))

	# Zip the arrays together
	paired_arrays = list(zip(peaks, turn_idxs))
	# Sort the paired arrays based on the first array
	sorted_paired_arrays = sorted(paired_arrays, key=lambda x: x[0])
	# Unzip the sorted paired arrays
	if len(sorted_paired_arrays) == 0:
		return [], []
	peaks, turn_idxs = map(list, zip(*sorted_paired_arrays))

	return peaks, turn_idxs

def get_turns(ht=np.pi/3, output_file='test.output'):
	
	all_turns = []
	all_peak_times = []
	angle1 = []
	angle2 = []
	turn_lengths = []
	in_box_angles = []

	f1 = open(output_file, 'rb')
	bbv, data = pickle.load(f1)
	f1.close()

	for i in range(len(data)):
		
		sol, vels = data[i]

		pos = sol[:, 0:2]
		angles = sol[:, 2]

		speed = np.sqrt((vels[:,0])**2 + (vels[:,1])**2)
		angVels = vels[:, 2]

		reprocess_angles = (angles + np.pi)%(2*np.pi) - np.pi

		# To be used for normalizing to amount of time spent facing each direction
		in_box_angles.append(reprocess_angles)

		# Find turns using angVels
		peaks, turn_idxs = find_turn_indices(angVels, ht = ht)

		for k,turn_idx in enumerate(turn_idxs):
			# conditional to make sure indices dont go out
			turn0, turn1 = turn_idx[0], turn_idx[1]
			
			# Get difference of ingoing vs outgoing angle
			turn_angle = angles[turn1] - angles[turn0]

			all_peak_times.append(peaks[k]*bbv.dt)
			angle1.append(reprocess_angles[turn0])
			angle2.append(reprocess_angles[turn1])
			all_turns.append(turn_angle)
			turn_lengths.append((turn1-turn0)*bbv.dt)

	return all_turns, all_peak_times, angle1, angle2, turn_lengths, in_box_angles

def turn_distribution(ht=np.pi/3, plot_name='test', mode ='angle', side=False, output_file = 'test.output'):

	outputDir = f'plots/{plot_name}/'
	create_directory(outputDir)

	all_turns, all_peak_times, angle1, angle2, turn_lengths, in_box_angles = get_turns(ht=ht, output_file=output_file)

	fig, ax = plt.subplots()

	if mode ==  'angle':
		q1, q2 = 25, 75
		# Conversion to deg
		all_turns = (180/np.pi)*np.array(all_turns)

		# Plot the distribution of "turns"
		if side == False:
			all_turns = np.abs(all_turns)

		# print(len(all_turns),all_turns)
		sns.histplot(x=all_turns, ax=ax)

		ax.set_xlabel('Angle (deg)')
		xlim_max = 180
		if side == False:
			ax.set_xticks(30*(np.arange(6)))
			ax.set_xlim([0, xlim_max])
		else:
			ax.set_xticks(30*(np.arange(11)-5))
			ax.set_xlim([-xlim_max, xlim_max])

		title = f'AngVel threshold: {np.round((180/np.pi)*ht)} deg/s, Percentiles are {q1}%: {np.round(np.percentile(all_turns, q1))}, {q2}%: {np.round(np.percentile(all_turns, q2))}'
	
	elif mode == 'time':
		q1, q2 = 25, 75
		sns.histplot(x=turn_lengths, kde=True, ax=ax)
		ax.set_xlabel('Time spent in turn (seconds)')
		title = f'AngVel threshold: {np.round((180/np.pi)*ht)} deg/s, Percentiles are {q1}%: {np.round(np.percentile(turn_lengths, q1),2)}, {q2}%: {np.round(np.percentile(turn_lengths, q2), 2)} seconds'


	ax.set_title(title)

	fig.suptitle(plot_name)
	fig.tight_layout()
	
	fig.savefig(outputDir+'turn_distribution_'+mode+'.png', transparent=True)

	fig.clf()
	plt.close(fig)

def distribution_in_out(ht=np.pi/3, plot_name='test', output_file = 'test.output'):

	outputDir = f'plots/{plot_name}/'
	create_directory(outputDir)

	all_turns, all_peak_times, angle1, angle2, turn_lengths, in_box_angles = get_turns(ht=ht, output_file=output_file)
			
	# Plot to see conditional distribution
	nbins = 6
	bins = np.linspace(-np.pi, np.pi, nbins+1)

	hist, _ = np.histogram(angle1, bins=bins)
	p1 = hist/len(angle1)
	hist, _ = np.histogram(angle2, bins=bins)
	p2 = hist/len(angle2)

	hist, _, _ = np.histogram2d(angle1, angle2, bins=bins)
	p12 = hist/len(angle1)
	
	# Calculate conditional distribution P(angle2 | angle1)
	p_angle2_given_angle1 = p12 / p1[:, None]

	fig, ax = plt.subplots()
	imshow_plot = ax.imshow(p_angle2_given_angle1)

	# Add numbers on the imshow plot
	for i in range(nbins):
		for j in range(nbins):
			ax.text(j, i, f'{p_angle2_given_angle1[i, j]:.2f}', ha='center', va='center', color='white')

	# Set xticks and labels
	ticks = np.linspace(-0.5, nbins-1+0.5, nbins+1)
	ticklabels = np.round(np.linspace(-180, 180, nbins+1))	# cast angles to deg
	# ticklabels = np.round(np.linspace(-np.pi, np.pi, nbins+1), 2)

	# Add lines to middle
	for y in ticks:
		ax.axhline(y, color='red', linestyle='--')

	fig.gca().invert_yaxis()
	ax.set(xticks=ticks, xticklabels=ticklabels)
	ax.set(yticks=ticks, yticklabels=ticklabels)
	ax.set(xlabel='angle_next', ylabel='angle_prev')
	ax.set(title = 'P(angle_next | angle_prev)')

	# Add a colorbar to the plot
	colorbar = plt.colorbar(imshow_plot)
	imshow_plot.set_clim(vmin=0, vmax=0.5)

	fig.suptitle(plot_name)
	fig.tight_layout()

	# Save plot
	fig.savefig(outputDir+'turn_conditionals.png', transparent=True)
	
	fig.clf()
	plt.close(fig)

def joyplot_in_out(ht=np.pi/3, plot_name='test', output_file = 'test.output'):
	
	outputDir = f'plots/{plot_name}/'
	create_directory(outputDir)

	all_turns, all_peak_times, angle1, angle2, turn_lengths, in_box_angles = get_turns(ht=ht, output_file=output_file)

	angle1 = np.array(angle1)
	# angle2_reprocess = (angle2 - mid_angles[inds-1] + np.pi)%(2*np.pi) - np.pi
	# angle_diff = (angle2 - angle1 + np.pi)%(2*np.pi) - np.pi
	# angle_diff_reprocess = (180/np.pi)*angle_diff # convert to deg
	angle_diff_reprocess = (180/np.pi)*np.array(all_turns)

			
	# Plot to see conditional distribution
	nbins = 6
	bins = np.linspace(-np.pi, np.pi, nbins+1)
	mid_angles = (bins[1:] + bins[:-1])/2
	inds = np.digitize(angle1, bins)

	df = pd.DataFrame({'Angle1': angle1, 'Angle_diff': angle_diff_reprocess , 'inds_Angle1': inds})

	# fig, axs = joypy.joyplot(df, by='inds_Angle1', column='Angle2', overlap=0, figsize=(10, 6), hist=True, density=True, bins=np.linspace(-180, 180, nbins*10+1))
	# fig, axs = joypy.joyplot(df, by='inds_Angle1', column='Angle2', overlap=0, figsize=(10, 6))

	fig, axs = plt.subplots(nbins, 2, figsize=(10,6))

	for k in range(len(axs)):
		df_subset = df[df.inds_Angle1 == (nbins-k)]
		sns.histplot(data=df_subset, x="Angle_diff", element='step', stat='density', bins=np.linspace(-180, 180, nbins*4+1), ax = axs[k][1])
		sns.kdeplot(data=df_subset, x="Angle_diff", fill=True, ax = axs[k][0])
		ang2 = int(np.round((180/np.pi)*(bins[nbins-k])))
		ang1 = int(np.round((180/np.pi)*(bins[nbins-k-1])))

		axs[k][0].set_ylabel(f'[{ang1},{ang2}]')
		axs[k][1].set_ylabel('')
		
		for i in range(2):
			axs[k][i].set_xlabel('')
			axs[k][i].set_xlim([-180, 180])

			axs[k][i].spines['top'].set_visible(False)
			axs[k][i].spines['right'].set_visible(False)
			axs[k][i].spines['left'].set_visible(False)

	max_value = max(ax.get_ylim()[1] for ax in [axs[k][0] for k in range(nbins)])
	max_value2 = max(ax.get_ylim()[1] for ax in [axs[k][1] for k in range(nbins)])
	for k in range(len(axs)):
		axs[k][0].set_ylim([0, max_value])
		axs[k][1].set_ylim([0, max_value2])

	fig.suptitle(plot_name)
	fig.tight_layout()

	fig.savefig(outputDir+'turn_joyplot.png', transparent=True)
	# fig.savefig(outputDir_png + 'turn_joyplot_'+groupName+'.png')
		
	fig.clf()
	plt.close('all')

def polar_turns(ht=np.pi/3, plot_name='test', mode = None, output_file = 'test.output'):

	outputDir = f'plots/{plot_name}/'
	create_directory(outputDir)

	all_turns, all_peak_times, angle1, angle2, turn_lengths, in_box_angles = get_turns(ht=ht, output_file=output_file)

	angle1 = np.array(angle1)
	angle_diff_reprocess = (180/np.pi)*np.array(all_turns)

	# Plot to see conditional distribution
	nbins = 8
	bins = np.linspace(-np.pi, np.pi, nbins+1)
	mid_angles = (bins[1:] + bins[:-1])/2
	inds = np.digitize(angle1, bins)
 
	min_alpha = 0.2
	# alphas = [min_alpha, 1, min_alpha, 1]
	alphas = [min_alpha, 1,1, min_alpha,min_alpha, 1, 1, min_alpha, ]

	df = pd.DataFrame({'Angle1': angle1, 'Angle_diff': angle_diff_reprocess , 'inds_Angle1': inds, 'peaks': all_peak_times})

	if mode is None:
		dfs = [df]
		fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6,6))
		ax = [ax]

	elif mode  == 'stratify':
		bdry = 60
		dfs = [df[df.peaks < bdry], df[df.peaks >= bdry]]
		fig, ax = plt.subplots(1,2,subplot_kw={'projection': 'polar'}, figsize=(12,6))
		ax[0].set_title(f'First {bdry} seconds')
		ax[1].set_title(f'After {bdry} seconds')

	for i,df in enumerate(dfs):
		# print(len(df))
		lCountNorm, rCountNorm = np.zeros(nbins), np.zeros(nbins)
		for k in range(nbins):
			df_subset = df[df.inds_Angle1 == (k+1)]
			pos = df_subset[df_subset['Angle_diff'] > 0].shape[0]
			neg = df_subset[df_subset['Angle_diff'] < 0].shape[0]
			lCountNorm[k] = pos/(pos+neg)
			rCountNorm[k] = neg/(pos+neg)

		# p1 = ax[i].bar(mid_angles,rCountNorm,width = (2 * np.pi / nbins), color='purple',edgecolor='k',linewidth=1.5)
		# p3 = ax[i].bar(mid_angles,lCountNorm,width = (2 * np.pi / nbins), color='green', bottom =rCountNorm,edgecolor='k',linewidth=1.5)
  
		p1 = ax[i].bar(mid_angles,rCountNorm,width = (2 * np.pi / nbins), color=list(zip(['purple']*nbins, alphas)),edgecolor=list(zip(['black']*nbins, alphas)),linewidth=1.5)
		p3 = ax[i].bar(mid_angles,lCountNorm,width = (2 * np.pi / nbins), color=list(zip(['green']*nbins, alphas)), bottom =rCountNorm, edgecolor=list(zip(['black']*nbins, alphas)),linewidth=1.5)

		ax[i].legend((p1[0],p3[0]),('Right','Left'),loc='upper right')
		ax[i].set_xlabel('Incoming angle')
		ax[i].set_rorigin(-1.0)
		ax[i].set_ylim([0,1])
		ax[i].xaxis.grid(False)
		ax[i].set_aspect('equal')

	fig.suptitle(plot_name)
	fig.tight_layout()

	fig.savefig(outputDir+'polar_turns.png')
	# fig.savefig(outputDir_png + default +groupName+'.png')

def num_turns_direction(ht=np.pi/3, plot_name='test', output_file = 'test.output'):
	
	outputDir = f'plots/{plot_name}/'
	create_directory(outputDir)

	all_turns, all_peak_times, angle1, angle2, turn_lengths, in_box_angles = get_turns(ht=ht, output_file=output_file)

	# Plot number of turns
	in_box_angles = np.concatenate(in_box_angles)
	angle1 = np.array(angle1)

	# print(np.min(in_box_angles), np.max(in_box_angles))
	# print(np.min(angle1), np.max(angle1))

	fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

	num_bins = 6
	colors = ['brown','grey','green','green','grey','brown']
	hist, bins = np.histogram(angle1, bins=num_bins, range=(-np.pi, np.pi))
	inds = np.digitize(angle1, bins)
	inds2 = np.digitize(in_box_angles, bins)
	
	mag = np.array([np.sum(angle1[inds==(i+1)])/np.sum(in_box_angles[inds2==(i+1)]) for i in range(num_bins)])

	barbins = bins[:-1] + np.pi / num_bins
	ax.bar(barbins, mag, width=0.75*(2 * np.pi / num_bins), align="center", color=colors, edgecolor='k')

	# ax.set_ylim([0,speed_threshold[1]/2])
	# ax.set_yticks(np.linspace(0,speed_threshold[1]/2,6))
	ax.set_title(f'Number of turns per second spent moving in each bin')
	
	# Save plot
	fig.suptitle(plot_name)
	fig.tight_layout()
	
	fig.savefig(outputDir+'num_turns_direction.png')
	# fig.savefig(outputDir_png + 'num_turns_direction_'+groupName+'.png')

	fig.clf()
	plt.close(fig)

def percent_reached_plot(plot_name='test', output_file = 'test.output'):

	outputDir = f'plots/{plot_name}/'
	create_directory(outputDir)

	lineDists = [0.2,0.4,0.6,0.8]
	fig, ax = plt.subplots()
	allLineInds = []
	allCumDists = []

	f1 = open(output_file, 'rb')
	bbv, data = pickle.load(f1)
	f1.close()

	for i in range(len(data)):
		
		sol, vels = data[i]

		pos = sol[:, :2]/10
		angles = sol[:, 2]

		speed = np.sqrt((vels[:,0])**2 + (vels[:,1])**2)
		angVels = vels[:, 2]

		normalized_x = sol[:,0]/bbv.stageW

		lineInds = []
		for l in lineDists:
			ind = next((i for i in range(len(pos)) if normalized_x[i] > l), None)
			lineInds.append((ind) if ind is not None else None)
		
		distances = np.linalg.norm(pos[1:,] - pos[:-1,], axis=1)
		cumulative_distances = np.cumsum(distances)
		# lineFirstHitDist = [cumulative_distances[i-1] if i is not None else None for i in lineInds]
		lineFirstHitDist = [cumulative_distances[i-1] if i is not None else cumulative_distances[-1] for i in lineInds]

		allLineInds.append(lineInds)
		allCumDists.append(lineFirstHitDist)

	percent_reached = []
	distances_reached = []
	med_dist_reached = []

	numFiles = len(allLineInds)
	for i in range(len(lineDists)):
		count = 0
		dists = []
		for j in range(numFiles):
			if allLineInds[j][i] is not None:
				count += 1
			dists.append(allCumDists[j][i])

		percent_reached.append(count/numFiles)
		distances_reached.append(dists)
		med_dist_reached.append(np.median(dists))

	ax.boxplot(distances_reached)
	for i,list in enumerate(distances_reached):
		plt.scatter([i+1]*len(list), list)
	ax.set_xticklabels([round(p, 2) for p in percent_reached])
	ax.set_ylabel('Distance walked (cm)')
	ax.set_ylim([0, 300])

	# Save plot
	fig.suptitle(plot_name)
	fig.tight_layout()

	fig.savefig(outputDir+'dist_reached.png')
	plt.close()

def average_velocity(plot_name = 'test', output_file = 'test.output'):

	outputDir = f'plots/{plot_name}/'
	create_directory(outputDir)

	all_angles = []
	all_angvels = []
	all_speeds = []

	f1 = open(output_file, 'rb')
	bbv, data = pickle.load(f1)
	f1.close()

	for i in range(len(data)):
		
		sol, vels = data[i]

		pos = sol[:, :2]
		angles = sol[:, 2]

		speed = np.sqrt((vels[:,0])**2 + (vels[:,1])**2)
		angVels = vels[:, 2]

		all_angles.append(sol[:, 2])
		all_angvels.append(angVels)
		all_speeds.append(speed)

	all_angles = np.concatenate(all_angles)
	all_angvels = np.abs(np.concatenate(all_angvels))
	all_speeds = np.concatenate(all_speeds)

	# velocity plot
	fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
	num_bins = 6
	colors = ['cyan','grey','pink','pink','grey','cyan']
	hist, bins = np.histogram(all_angles, bins=num_bins, range=(0, 2 * np.pi))
	inds = np.digitize(all_angles, bins)
	# calculate the average speed for each angle bin
	mag = np.array([np.average(all_speeds[inds==(i+1)]) for i in range(num_bins)])
	barbins = bins[:-1] + np.pi / num_bins
	ax.bar(barbins, mag, width=0.75*(2 * np.pi / num_bins), align="center", color=colors, edgecolor='k')
	# ax.set_ylim([0,10])
	ax.set_title('Average speed (mm/s)')
	
	# Save plot
	fig.suptitle(plot_name)
	fig.tight_layout()
	
	fig.savefig(outputDir+'velo_plot.png')
	plt.close('all')

	# angvel plot
	fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
	num_bins = 6
	colors = ['green','grey','brown','brown','grey','green']
	hist, bins = np.histogram(all_angles, bins=num_bins, range=(0, 2 * np.pi))
	inds = np.digitize(all_angles, bins)
	# calculate the average angvel for each angle bin
	mag = (180/np.pi)*np.array([np.average(all_angvels[inds==(i+1)]) for i in range(num_bins)])
	barbins = bins[:-1] + np.pi / num_bins
	ax.bar(barbins, mag, width=0.75*(2 * np.pi / num_bins), align="center", color=colors, edgecolor='k')
	# ax.set_ylim([0,5])
	ax.set_title('Average angular velocity (deg/s)')
 
 	# Save plot
	fig.suptitle(plot_name)
	fig.tight_layout()

	fig.savefig(outputDir+'angvel_plot.png', transparent=True)
	plt.close('all')

def sample_plot(plot_name = 'test', output_file = 'test.output', ht = np.pi/2, num_plots = 10):

	outputDir = f'plots/{plot_name}/'
	create_directory(outputDir)
	create_directory(outputDir+'samples/')

	f1 = open(output_file, 'rb')
	bbv, data = pickle.load(f1)
	f1.close()

	for i in range(num_plots):
		
		sol, vels = data[i]

		angles = sol[:, 2]
		angVels = vels[:, 2]

		sol = sol/10

		peaks, turn_idxs = find_turn_indices(angVels, ht = ht)
		turn_angles = np.array([angles[t[1]] - angles[t[0]] for t in turn_idxs])

		fig, ax = plt.subplots()
		ax.scatter(sol[:,0], sol[:,1], s=0.5, color='k')
		ax.plot(sol[:,0], sol[:,1], linewidth=0.1)
		ax.set(xlim=[0, bbv.stageW/10], ylim=[0, bbv.stageH/10])
		ax.imshow(t0,extent=[0,bbv.stageW/10,0,bbv.stageH/10],cmap=cm1,vmin=25,vmax=40.)

		for k,turn_idx in enumerate(turn_idxs):
			# conditional to make sure indices dont go out
			turn0, turn1 = turn_idx[0], turn_idx[1]
			pos_segment = sol[turn0:turn1,:]
	
			if turn_angles[k] >= 0:
				ax.plot(pos_segment[:,0],pos_segment[:,1],linewidth=0.3, zorder=20, color='yellow')
			else:
				ax.plot(pos_segment[:,0],pos_segment[:,1],linewidth=0.3, zorder=20, color='cyan')

		ax.set_aspect('equal')
		ax.set_title(f'Sample plot {i}')
		fig.savefig(f'{outputDir}samples/sample_{i}.pdf', transparent=True)
		plt.close()
