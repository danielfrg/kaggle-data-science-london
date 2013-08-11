#!/usr/bin/env python

import numpy as np
from sklearn import grid_search
from sklearn import cross_validation as cv
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.decomposition import PCA

loadData = lambda f: np.genfromtxt(open(f,'r'), delimiter=',')


pca = PCA(n_components=12,whiten=True)

test = pca.fit_transform(loadData('test.csv'))
train = pca.transform(loadData('train_inputs.csv'))
target = loadData('train_labels.csv')

#gamma_range = 10 ** np.arange(-4,-1,1)
#C_range = 10.0 ** np.arange(7,-1,-1)
#params = dict(gamma=gamma_range,C=C_range)

cvk = StratifiedKFold(target,k=3)
params = dict(gamma=[0.277777777778],C=[1000000])
classifier = SVC()


clf = grid_search.GridSearchCV(classifier,param_grid=params,cv=cvk)
clf.fit(train,target)
print("The best classifier is: ",clf.best_estimator_)

# Estimate score
scores = cv.cross_val_score(clf.best_estimator_, train, target, cv=60)
print('Estimated score: %0.5f (+/- %0.5f)' % (scores.mean(), scores.std() / 2))

# Predict and save
result = clf.best_estimator_.predict(test)

f=open('result.csv','w')
f.write('Id,Solution\n')

count=1

for x in result:
    f.write('%d,%d\n' % (count,x))
    count += 1

f.close()




