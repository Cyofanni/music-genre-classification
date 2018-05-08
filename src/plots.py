from matplotlib import pyplot as plt


def plot_confusion_matrix(cm, genre_list, name, title):
	plt.clf()
	plt.matshow(cm, fignum=False, cmap='Blues', vmin=0, vmax=1.0)
	ax = plt.axes()
	ax.set_xticks(range(len(genre_list)))
	ax.set_xticklabels(genre_list)
	ax.xaxis.set_ticks_position("bottom")
	ax.set_yticks(range(len(genre_list)))
	ax.set_yticklabels(genre_list)
	plt.title(title)
	plt.colorbar()
	plt.grid(False)
	plt.xlabel('Predicted class')
	plt.ylabel('True class')
	plt.grid(False)
	plt.savefig('ConfusionMatrix')
	plt.show()

def plot_roc_curves(fpr, tpr, AUC, label):
	f, axarr = plt.subplots(3, 3, figsize=(14, 12))
	plt.subplots_adjust(left=0.1, right=0.925, top=0.925, bottom=0.1, wspace=0.3, hspace=0.5)

	for r in range(3):
		for c in range(3):
			i = 3 * r + c

			axarr[r][c].plot(fpr[i], tpr[i], 'lightblue')
			axarr[r][c].fill_between(fpr[i], 0, tpr[i], facecolor='lightblue', alpha=0.5)
			axarr[r][c].plot([0, 1], [0, 1], 'b--')
			axarr[r][c].set_xlim([0.0, 1.0])
			axarr[r][c].set_ylim([0.0, 1.0])
			axarr[r][c].set_xlabel('False Positive Rate')
			axarr[r][c].set_ylabel('True Positive Rate')
			axarr[r][c].set_title('ROC curve for ' + label[i] + ' - AUC = %0.2f' % AUC[i])
			plt.savefig('ROCcurve')

	plt.show()
