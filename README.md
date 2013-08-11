kaggle-data-science-london
==========================

What I learned:

The most important insight was the semi-supervised approach described [here](http://www.kaggle.com/c/data-science-london-scikit-learn/forums/t/4986/ideas-that-worked-or-did-not). The basic idea is:

>the general idea is to predict the labels of the test corpus and use them as part of the training corpus. One needs to be careful though.

>Just to be sure then : build a model using the training data. Use this model to predict the test set labels. Combine the training set with these test set label predictions, along with the test set features to create a new training set of 10,000 observations

I changed the idea a little bit using only good predictions (>95%) and iterated a few times.

The over-fitting was a problem a big C seamed to fix it.

Multicollinearity was a problem but it was solved by PCA. There is difference between PCA with and without whiten, on this case whiten=True was better.

Once again I learned the importance of cross-validation. I spent days using a simple train_test_split
and had bad luck, I was getting improvements but could not see them because of that particular split.
USE CV! :P

The Neural Networks failed, not sure why but a "simple" SVM was very good from the beginning.