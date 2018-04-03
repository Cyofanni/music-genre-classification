from scikits.talkbox.features import mfcc
import scipy
from scipy.io import wavfile
import numpy as np
import os
import glob


def write_ceps(ceps, filename):
	base_filename, ext = os.path.splitext(filename)
	data_filename = base_filename + ".ceps"
	np.save(data_filename, ceps)
	print("Written %s" % data_filename)

def create_ceps(fn):
	s_rate, X = scipy.io.wavfile.read(fn)
	ceps, mspec, spec = mfcc(X)
	write_ceps(ceps, fn)

def read_ceps(genre_list, base_dir):
	X, y = [], []
	for l, g in enumerate(genre_list):
		for fn in glob.glob(os.path.join(base_dir, g, "*.ceps.npy")):
			ceps = np.load(fn)
			num_ceps = len(ceps)
			X.append(np.mean(ceps[int(num_ceps*1/10):int(num_ceps*9/10)], axis=0))
			#X.append(np.mean(ceps, axis=0))   #doesn't help, it only increases running time
			y.append(l)

	return np.array(X), np.array(y)

