import scipy
from scipy.io import wavfile
import os
import numpy as np
import glob
from matplotlib import pyplot


GENRES = ["blues", "classical", "country", "disco", "hiphop", "jazz",
	  "metal", "pop", "reggae", "rock"]

def create_fft(filename):
	s_rate, data = scipy.io.wavfile.read(filename)   #get sample rate and data  
	fft_feat = abs(scipy.fft(data)[:1000])
	#fft_features = abs(scipy.fft(data)[:3000])  #doesn't help, it only increases running time
	base_filename, ext = os.path.splitext(filename)
	data_filename = base_filename + ".fft"
	np.save(data_filename, fft_feat)

def read_fft(genre_list, base_dir):
	X = []     #will store fft features
	y = []     #will store labels
	for l, g in enumerate(genre_list):    #loop using label as numeric index, and genre as iterator over genres
	 	genre_dir = os.path.join(base_dir, g, "*.fft.npy")   #create something like: "genres/classical/*.fft.npy"
		file_list = glob.glob(genre_dir)     #retrieve al the files under "genres/classical" with "fft.npy" ext
		for f in file_list:
			fft_features = np.load(f)
			X.append(fft_features[:1000])
			#X.append(fft_features[:3000]) ##doesn't help, it only increases running time
			y.append(l)
	return np.array(X), np.array(y)





#def plot_confusion_matrix(cm, genre_list, name, title):
#	pyplot.clf()
#	pyplot.matshow(cm, fignum=False, cmap='Blues', vmin=0, vmax=1.0)
#	ax = pyplot.axes()
#	ax.set_xticks(range(len(genre_list)))
#	ax.set_xticklabels(genre_list)
#	ax.xaxis.set_ticks_position("bottom")
#	ax.set_yticks(range(len(genre_list)))
#	ax.set_yticklabels(genre_list)
#	pyplot.title(title)
#	pyplot.colorbar()
#	pyplot.grid(False)
#	pyplot.xlabel('Predicted class')
#	pyplot.ylabel('True class')
#	pyplot.grid(False)
#	pyplot.show()


#X, y = read_fft(GENRES, GENRE_DIR)      #get the data set

#TEST GRAPHICAL CONFUSION MATRIX
#from sklearn.metrics import confusion_matrix
#from sklearn.preprocessing import normalize
#y_true = [1,1,1,2,2,2,3,3,3,4,4,4]
#y_pred = [1,2,1,3,3,2,3,3,3,4,3,2]
#cm = confusion_matrix(y_true, y_pred)
#cm = normalize(cm)   #remember to normalize
#plot_confusion_matrix(cm, ["classical","blues","jazz","country"], "name", "plot")


