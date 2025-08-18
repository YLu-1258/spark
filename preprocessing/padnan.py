import numpy as np

def fill_nan_with_zero(features):
	"""
	Fill NaN values in the features array with zeros.

	Parameters:
	- features: 2D numpy array of shape (num_features, num_windows)

	Returns:
	- features_filled: 2D numpy array with NaNs replaced by zeros
	"""
	return np.nan_to_num(features, nan=0.0)	
