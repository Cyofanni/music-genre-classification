from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from music_genre_mfcc import read_ceps
from plots import plot_confusion_matrix
from plots import plot_roc_curves

import numpy as np


genre_list = ["classical", "jazz", "country", "pop", "rock", "metal"]


if __name__ == "__main__":
	# reading inputs
	x, y = read_ceps(genre_list, "genres")

	# logistic regression train and prediction
	lr = LogisticRegression()
	lr.fit(x, y)
	y_pred = lr.predict(x)

	# confusion matrix
	cm = normalize(confusion_matrix(y, y_pred), axis=1, norm='l1')
	plot_confusion_matrix(cm, genre_list, "Name", "Confusion Matrix")

	# ROC curve
	fprs = []
	tprs = []
	AUCs = []

	for label in range(len(genre_list)):
		y_label = np.asarray(y==label, dtype=int)
		proba = lr.predict_proba(x)
		proba_label = proba[:,label]
		fpr, tpr, roc_thresholds = roc_curve(y_label, proba_label)
		fprs.append(fpr)
		tprs.append(tpr)
		AUCs.append(roc_auc_score(y_label, proba_label))

	plot_roc_curves(fprs, tprs, AUCs, genre_list)