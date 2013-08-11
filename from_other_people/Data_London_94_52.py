import numpy as np
from sklearn import grid_search
from sklearn import cross_validation as cv
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cross_validation import StratifiedKFold

workDir = r''

# Read data
train = np.genfromtxt(open(workDir + 'train_inputs.csv','rb'), delimiter=',')
target = np.genfromtxt(open(workDir + 'train_labels.csv','rb'), delimiter=',')
test = np.genfromtxt(open(workDir + 'test.csv','rb'), delimiter=',')

pca = PCA(n_components=12,whiten=True)
train = pca.fit_transform(train)
test = pca.transform(test)

C_range = 10.0 ** np.arange(7,10)
gamma_range = 10.0 ** np.arange(-4,0)
params = dict(gamma=gamma_range,C=C_range)
cvk = StratifiedKFold(target,k=3)
classifier = SVC()
clf = grid_search.GridSearchCV(classifier,param_grid=params,cv=cvk)
clf.fit(train,target)
print("The best classifier is: ",clf.best_estimator_)

C_range = 10.0 ** np.arange(6.5,7.5,.25)
gamma_range = 10.0 ** np.arange(-1.5,0.5,.25)
params = dict(gamma=gamma_range,C=C_range)
cvk = StratifiedKFold(target,k=3)
classifier = SVC()
clf = grid_search.GridSearchCV(classifier,param_grid=params,cv=cvk)
clf.fit(train,target)
print("The best classifier is: ",clf.best_estimator_)


# Estimate score
scores = cv.cross_val_score(clf.best_estimator_, train, target, cv=30)
print('Estimated score: %0.5f (+/- %0.5f)' % (scores.mean(), scores.std() / 2))

# Predict and save
result = clf.best_estimator_.predict(test)

np.savetxt(workDir + 'result.csv', result, fmt='%d')

