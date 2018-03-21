from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize

from music_genre_mfcc import read_ceps
from music_genre_fft import plot_confusion_matrix


genre_list = ["classical", "jazz", "country", "pop", "rock", "metal"]


if __name__ == "__main__":
	x, y = read_ceps(genre_list, "genres")

	lr = LogisticRegression()
	lr.fit(x, y)
	y_pred = lr.predict(x)

	cm = normalize(confusion_matrix(y, y_pred), axis=1, norm='l1')
	print(cm)
	plot_confusion_matrix(cm, genre_list, "Name", "Confusion Matrix")