import numpy as np
import math
import os
import itertools
from scipy.signal import medfilt, savgol_filter
from scipy.ndimage import gaussian_filter1d


class MemoryBuffer:
	def __init__(self, capacity):
		self.capacity = capacity
		self.buffer = []

	def push(self, x):
		if len(self.buffer) >= self.capacity:
			self.buffer.pop(0)
		self.buffer.append(x)

	def arrayize(self):
		return np.array(self.buffer)

	def __len__(self):
		return len(self.buffer)


def angle_reflect(angle, reflect_angle):
	my_angle = angle - 2*(angle - reflect_angle)
	return math.remainder(my_angle, 2*math.pi)

def angle_diff(theta1, theta2):
	diff = np.abs(theta1 - theta2)%(2*np.pi)

	angle_diff = diff + (diff>np.pi)*(2*np.pi - 2*diff)
	# angle_diff = np.pi - np.abs(np.pi-diff)

	return angle_diff

def indices_grouped_by_condition(array, condition):

	# Using enumerate to get indices and values
	enumerated_data = list(enumerate(array))
	# Using groupby to group consecutive elements based on the condition
	grouped_data = itertools.groupby(enumerated_data, key=lambda x: condition(x[1]))
	# Filtering out groups where the condition is not met and extracting element & indices
	result = [list(indices) for condition_met, indices in grouped_data if condition_met]
	# Filtering out indices only since each element is (item, index)
	result = [[x[0] for x in segment] for segment in result]

	return result

def create_directory(outputDir):
    CHECK_FOLDER = os.path.isdir(outputDir)
    if not CHECK_FOLDER:
        os.makedirs(outputDir)
        print("Created folder: ", outputDir)

def is_number_in_interval(array, interval):
    # Check if any number in the array falls within the specified interval.
    # Parameters:
    #     array (numpy.ndarray): Input array of numerical values.
    #     interval (tuple): Tuple containing the lower and upper bounds of the interval.
    # Returns:
    #     bool: True if any number falls within the interval, False otherwise.
    lower_bound, upper_bound = interval
    return np.any((array >= lower_bound) & (array <= upper_bound))

def smooth_and_deriv(x, dt, window_length = 9, polyorder = 3, sigma = 2):
	xhat = savgol_filter(x, window_length=window_length, polyorder=polyorder) 
	dxdt = savgol_filter(x, window_length=window_length, polyorder=polyorder, deriv = 1) / dt
	dxdt = gaussian_filter1d(dxdt, sigma = sigma, mode='nearest')
	return xhat, dxdt

def calc_direction(v, x):
	xmean = np.mean(v * np.cos(x))
	ymean = np.mean(v * np.sin(x))
	return np.arctan2(ymean, xmean)

def cast_to(theta):
	return (theta+ np.pi)%(2*np.pi) - np.pi

def ReLU(x):
    return x * (x > 0)

def ELU(x):
	return np.where(x >= 0, x, np.exp(x) - 1)

