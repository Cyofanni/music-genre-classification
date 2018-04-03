import music_genre_fft as mgfft
import music_genre_mfcc as mgmfcc
import os

GENRE_DIR = "../genres"

def genre_create_fftandceps(direct):      #create fft and ceps files for a given directory (direct for a genre)
    for filename in os.listdir(direct):
	if filename.endswith(".wav"):
		mgfft.create_fft(direct+filename)
		mgmfcc.create_ceps(direct+filename)


def root_create_fftandceps(root):    #create fft and ceps files for every genre starting from the root ("genres")
    for subdir_name in os.listdir(root):
	if os.path.isdir(root + subdir_name):
	    genre_path = root + subdir_name
	    genre_create_fftandceps(genre_path + "/")


root_create_fftandceps(GENRE_DIR + "/")     #run the feature extraction for each folder

