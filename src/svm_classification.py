from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import scale
from sklearn.preprocessing import normalize
import sys
import numpy as np

import music_genre_fft as mg_fft
import music_genre_mfcc as mg_mfcc
import plots


genres_dir = '../genres'

def run_classifier(data, target):
   data_train, data_test, target_train, target_test = train_test_split(data, target)
   pline = Pipeline([('clf', SVC(kernel='rbf', gamma=0.01, C=100))])
   print "Shape of training set: %s" % (data_train.shape,)
   print "Shape of test set: %s" % (data_test.shape,)

   params = {
	   'clf__gamma': (0.01, 0.03, 0.1, 0.3, 1),
	   'clf__C': (0.1, 0.3, 1, 3, 10, 30),
   }

   gsearch = GridSearchCV(pline, params, n_jobs=2,
		   verbose=1, scoring='accuracy')

   gsearch.fit(data_train, target_train)
   print 'Best score: %0.3f' % gsearch.best_score_
   
   print 'Best parameters set:'
   best_params = gsearch.best_estimator_.get_params()
   
   for param_name in sorted(params.keys()):
	   print '\t%s: %r' % (param_name, best_params[param_name])

   preds = gsearch.predict(data_test) 
   print classification_report(target_test, preds)


if __name__ == '__main__':
   if (len(sys.argv)) < 2:
      print "Usage: \"python svm_classification.py -fft\" or \"python svm_classification.py -mfcc\""
      sys.exit(1)
    
   if (sys.argv[1] == '-fft'):
      X, y = mg_fft.read_fft(mg_fft.GENRES, genres_dir)  #read data
      X = normalize(X)
      run_classifier(X, y)
   elif (sys.argv[1] == '-mfcc'):
      X, y = mg_mfcc.read_ceps(mg_fft.GENRES, genres_dir)  #read data     
      X = np.nan_to_num(X)
      X = normalize(X)
      run_classifier(X, y)
   else:
      print "Illegal third argument",
