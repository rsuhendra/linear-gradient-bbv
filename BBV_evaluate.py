from BBV_gradient import *

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


class Evaluate_gradient():
	def __init__(self, bbv):
		self.bbv = bbv
	
	def generate_data(self, NUM = 50):
		data = []
		for n in range(NUM):
			sols, vels = self.bbv.simulate()
			data.append([sols, vels])
		return data

	def get_turns(self, data, ht=np.pi/3):
		
		all_turn_mags = []
		angle1 = []
		angle2 = []
		turn_durations = []
		in_box_angles = []

		for i in range(len(data)):
			
			sol, vels = data[i]
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

				all_turn_mags.append(turn_angle)
				angle1.append(reprocess_angles[turn0])
				angle2.append(reprocess_angles[turn1])
				turn_durations.append((turn1-turn0)*self.bbv.dt)

		return all_turn_mags, angle1, angle2, turn_durations, in_box_angles
	
	def polar_turns(self, turn_data):

		all_turn_mags, angle1, angle2, turn_durations, in_box_angles = turn_data

		angle1 = np.array(angle1)
		angle_diff_reprocess = (180/np.pi)*np.array(all_turn_mags)

		# Plot to see conditional distribution
		nbins = 8
		bins = np.linspace(-np.pi, np.pi, nbins+1)
		inds = np.digitize(angle1, bins)
	
		df = pd.DataFrame({'Angle1': angle1, 'Angle_diff': angle_diff_reprocess , 'inds_Angle1': inds})

		lCountNorm, rCountNorm = np.zeros(nbins), np.zeros(nbins)
		for k in range(nbins):
			df_subset = df[df.inds_Angle1 == (k+1)]
			pos = df_subset[df_subset['Angle_diff'] > 0].shape[0]
			neg = df_subset[df_subset['Angle_diff'] < 0].shape[0]
			lCountNorm[k] = pos/(pos+neg)
			rCountNorm[k] = neg/(pos+neg)

		return lCountNorm
 
	def num_turns_direction(self, turn_data):

		all_turn_mags, angle1, angle2, turn_durations, in_box_angles = turn_data

		# Plot number of turns
		in_box_angles = np.concatenate(in_box_angles)
		angle1 = np.array(angle1)
  
		num_bins = 6
		hist, bins = np.histogram(angle1, bins=num_bins, range=(-np.pi, np.pi))
		inds = np.digitize(angle1, bins)
		inds2 = np.digitize(in_box_angles, bins)
		
		mag = np.array([np.sum(angle1[inds==(i+1)])/np.sum(in_box_angles[inds2==(i+1)]) for i in range(num_bins)])

		return mag

	def distance_reached(self, data):
		
		lineDists = [0.2,0.4,0.6,0.8]
		allLineInds = []
		allCumDists = []
  
		for i in range(len(data)):
			sol, vels = data[i]

			pos = sol[:, :2]/10
			normalized_x = sol[:,0]/self.bbv.stageW

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
	
		return percent_reached

	def efficiency_measure(self, data):
     
		effx_list = []
		for i in range(len(data)):
			sol, vels = data[i]

			pos = sol[:, :2]/10
			normalized_x = sol[:,0]/self.bbv.stageW
	
			distances = np.linalg.norm(pos[1:,] - pos[:-1,], axis=1)
			cumulative_distances = np.cumsum(distances)

			limiter = 0.8

			ind = next((i for i in range(len(pos)) if normalized_x[i] > limiter), np.argmax(normalized_x))

			travel_to_max = cumulative_distances[ind-1]/self.bbv.stageW
			invade_dist = min(np.max(normalized_x), limiter)
			effx_list.append(invade_dist/travel_to_max)

		return np.median(effx_list)
 
	def binned_vel(self, data):
		
		all_angles = []
		all_angvels = []
		all_speeds = []
		
		for i in range(len(data)):
			sol, vels = data[i]

			speed = np.sqrt((vels[:,0])**2 + (vels[:,1])**2)
			angVels = vels[:, 2]

			all_angles.append(sol[:, 2])
			all_angvels.append(angVels)
			all_speeds.append(speed)

		all_angles = np.concatenate(all_angles)
		all_angvels = np.abs(np.concatenate(all_angvels))
		all_speeds = np.concatenate(all_speeds)

		# velocity plot
		num_bins = 6
		hist, bins = np.histogram(all_angles, bins=num_bins, range=(0, 2 * np.pi))
		inds = np.digitize(all_angles, bins)
		# calculate the average speed for each angle bin
		mag1 = np.array([np.average(all_speeds[inds==(i+1)]) for i in range(num_bins)])

		# angvel plot
		num_bins = 6
		hist, bins = np.histogram(all_angles, bins=num_bins, range=(0, 2 * np.pi))
		inds = np.digitize(all_angles, bins)
		# calculate the average angvel for each angle bin
		mag2 = (180/np.pi)*np.array([np.average(all_angvels[inds==(i+1)]) for i in range(num_bins)])

		return mag1, mag2
 
 
	def evaluate(self):
		data = self.generate_data()
		turn_data = self.get_turns(data, ht=np.pi/3)

		eval1 = self.efficiency_measure(data)
		eval2, eval3 = self.binned_vel(data)
		eval4 = self.polar_turns(turn_data)
  
		return eval1, eval2, eval3, eval4

def evaluate():
    pass