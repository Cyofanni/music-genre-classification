from scikits.talkbox.features import mfcc
import scipy
from scipy.io import wavfile
import numpy as np
import os
import glob

GENRE_DIR = "genres"

def write_ceps(ceps, fn):
	base_fn, ext = os.path.splitext(fn)
	data_fn = base_fn + ".ceps"
	np.save(data_fn, ceps)
	print("Written %s" % data_n)

def create_ceps(fn):
	sample_rate, X = scipy.io.wavfile.read(fn)
	ceps, mspec, spec = mfcc(X)
	write_ceps(ceps, fn)

def read_ceps(genre_list, base_dir=GENRE_DIR):
	X, y = [], []
	for label, genre in enumerate(genre_list):
		for fn in glob.glob(os.path.join(base_dir, genre, "*.ceps.npy")):
			ceps = np.load(fn)
			num_ceps = len(ceps)	
			X.append(np.mean(ceps[int(num_ceps*1/10):int(num_ceps*9/10)], axis=0))
			y.append(label)

	return np.array(X), np.array(y)
