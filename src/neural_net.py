from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from sklearn import model_selection
from sklearn.model_selection import cross_val_score, KFold, LeaveOneOut

from music_genre_mfcc import read_ceps
from music_genre_fft import read_fft
from plots import plot_confusion_matrix
from plots import plot_roc_curves

import numpy as np
import time
import sys


genre_list = ["blues", "classical", "country", "disco", "jazz", "metal", "pop", "reggae", "rock"]


if __name__ == "__main__":
	if (len(sys.argv)) < 2:
		print "Usage: \"python neural_net.py -fft\" or \"python neural_net.py -mfcc\""
		sys.exit(1)

	# reading inputs
	x = []
	y = []
	if (sys.argv[1] == '-fft'):
		x, y = read_fft(genre_list, "genres")
	elif (sys.argv[1] == '-mfcc'):
		x, y = read_ceps(genre_list, "genres")
	else:
		print "Options can only be -fft or -mfcc"
		sys.exit(1)

	start_time = time.time()

	# multi-layer perceptron model
	mlp = None
	if (sys.argv[1] == '-fft'):
		mlp = MLPClassifier(solver='adam')
	elif (sys.argv[1] == '-mfcc'):
		mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(13, 20, 20))
	print("Training, please wait")

	# cross validation
	folds = 100
	cv = KFold(n_splits=folds)
	scores = model_selection.cross_val_score(mlp, x, y, cv=cv)
	y_pred = model_selection.cross_val_predict(mlp, x, y, cv=cv)
	print("%1d-fold cross validation average accuracy: %.3f" % (folds, scores.mean()))

	print("--- %s seconds ---" % (time.time() - start_time))

	# classification report
	print(classification_report(y, y_pred))

	# confusion matrix
	cm = normalize(confusion_matrix(y, y_pred), axis=1, norm='l1')
	plot_confusion_matrix(cm, genre_list, "Name", "Confusion Matrix")

	print("Creating roc curves")

	# ROC curve
	fprs = []
	tprs = []
	AUCs = []

	for label in range(len(genre_list)):
		y_label = np.asarray(y==label, dtype=int)
		proba = model_selection.cross_val_predict(mlp, x, y, cv=cv, method='predict_proba')
		proba_label = proba[:,label]
		fpr, tpr, roc_thresholds = roc_curve(y_label, proba_label)
		fprs.append(fpr)
		tprs.append(tpr)
		AUCs.append(roc_auc_score(y_label, proba_label))

	plot_roc_curves(fprs, tprs, AUCs, genre_list)