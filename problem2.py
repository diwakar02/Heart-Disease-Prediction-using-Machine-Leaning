#This code performs the classification of heart disease by separating the predicted values
# in two sets, namely 0 for absence and 1 for presence where all the predicted values
# between 1 and 4 are replaced to 1 to check the model performance

from numpy import genfromtxt
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
import pylab as pl
from itertools import cycle
from sklearn import cross_validation
from sklearn.svm import SVC 

#Loading and pruning the data
dataset = genfromtxt('cleveland_data.csv',dtype = float, delimiter=',')
#print dataset
X = dataset[:,0:12] #Feature Set
y = dataset[:,13]   #Label Set

#Replacing 1-4 by 1 label
for index, item in enumerate(y):
    if not (item == 0.0):
	y[index] = 1
print y
target_names = ['0', '1']

#Method to plot the graph for reduced Dimesions
def plot_2D(data, target, target_names):
     colors = cycle('rgbcmykw')
     target_ids = range(len(target_names))
     plt.figure()
     for i, c, label in zip(target_ids, colors, target_names):
         plt.scatter(data[target == i, 0], data[target == i, 1],
                    c=c, label=label)
     plt.legend()
     plt.savefig('Problem 2 Graph')

# Classifying the data using a Linear SVM and predicting the probability of disease belonging to a particular class
modelSVM = LinearSVC(C=0.001)
pca = PCA(n_components=5, whiten=True).fit(X)
X_new = pca.transform(X)

# calling plot_2D
plot_2D(X_new, y, target_names)


#Applying cross validation on the training and test set for validating our Linear SVM Model
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_new, y, test_size=0.2, train_size=0.8, random_state=0)
modelSVM = modelSVM.fit(X_train, y_train)
print "Linear SVC values with split"
print modelSVM.score(X_test, y_test) 

modelSVMRaw = LinearSVC(C=0.001)
modelSVMRaw = modelSVMRaw.fit(X_new, y)
cnt = 0
for i in modelSVMRaw.predict(X_new):
	if i == y[i]:
	   cnt = cnt+1
print "Linear SVC score without split"
print float(cnt)/303

#Applying the Principal Component Analysis on the data features
modelSVM2 = SVC(C=0.001,kernel='rbf')

#Applying cross validation on the training and test set for validating our Linear SVM Model
X_train1, X_test1, y_train1, y_test1 = cross_validation.train_test_split(X_new, y, test_size=0.2, train_size=0.8, random_state=0)
modelSVM2 = modelSVM2.fit(X_train1, y_train1)
print "RBF score with split"
print modelSVM2.score(X_test1, y_test1)

modelSVM2Raw = SVC(C=0.001,kernel='rbf')
modelSVM2Raw = modelSVM2Raw.fit(X_new, y)
cnt1 = 0
for i in modelSVM2Raw.predict(X_new):
        if i == y[i]:
           cnt1 = cnt1+1
print "RBF score without split"
print float(cnt1)/303

#Using Stratified K Fold
skf = cross_validation.StratifiedKFold(y, n_folds=5)
for train_index, test_index in skf:
   # print("TRAIN:", train_index, "TEST:", test_index)
    X_train3, X_test3 = X[train_index], X[test_index]
    y_train3, y_test3 = y[train_index], y[test_index]
modelSVM3 = SVC(C=0.001,kernel='rbf')
modelSVM3 = modelSVM3.fit(X_train3, y_train3)
print "Stratified K fold score"
print modelSVM3.score(X_test3, y_test3)

modelSVM3Raw = SVC(C=0.001,kernel='rbf')
modelSVM3Raw = modelSVM3Raw.fit(X_new, y)
cnt2 = 0
for i in modelSVM3Raw.predict(X_new):
        if i == y[i]:
           cnt2 = cnt2+1
print "On PCA valued X_new"
print float(cnt2)/303


#change the value of n_component in variable pca and uncomment code below  to run the plot for PCA1 and PCA 2 values 
"""
# create a mesh to plot in
x_min, x_max = X_new[:, 0].min() - 1, X_new[:, 0].max() + 1
y_min, y_max = X_new[:, 1].min() - 1, X_new[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# title for the plots
titles = 'SVC (RBF kernel)- Plotting highest varied 2 PCA values'


# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
plt.subplot(2, 2, i + 1)
plt.subplots_adjust(wspace=0.4, hspace=0.4)
Z = modelSVM2.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
# Plot also the training points
plt.scatter(X_new[:, 0], X_new[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.title(titles)
plt.show()

"""
