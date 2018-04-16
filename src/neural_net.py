from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from sklearn import model_selection
from sklearn.model_selection import cross_val_score, KFold

from music_genre_mfcc import read_ceps
from plots import plot_confusion_matrix
from plots import plot_roc_curves

import numpy as np


genre_list = ["classical", "jazz", "country", "pop", "rock", "metal"]


if __name__ == "__main__":
	# reading inputs
	x, y = read_ceps(genre_list, "genres")

	# multi-layer perceptron model
	mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(13, 20, 20))
	print("Training, please wait")

	# cross validation
	folds = 100
	cv = KFold(n_splits=folds)
	scores = model_selection.cross_val_score(mlp, x, y, cv=cv)
	y_pred = model_selection.cross_val_predict(mlp, x, y, cv=cv)
	print("%1d-fold cross validation average accuracy: %.3f" % (folds, scores.mean()))

	# classification report
	print(classification_report(y, y_pred))

	# confusion matrix
	cm = normalize(confusion_matrix(y, y_pred), axis=1, norm='l1')
	plot_confusion_matrix(cm, genre_list, "Name", "Confusion Matrix")