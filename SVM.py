import pandas as pd
import numpy as np
from libsvm.svmutil import *

df = pd.read_csv("wbcd.csv")

df.head()
data = df.to_numpy()
data = np.delete(data,0,axis=1)
labels = data[:,0]

labels[labels=='M'] = 1
labels[labels=='B'] = 0
data = np.delete(data,0,axis=1)
data = np.delete(data,30,axis=1)

nb_train, nb_test = 400,169
x_train = data[:nb_train,:]
x_test = data[nb_train:,:]
y_train = labels[:nb_train]
y_test = labels[nb_train:]



##LINEAR KERNEL C=0.1
prob  = svm_problem(y_train, x_train)
param = svm_parameter('-t 0 -c 0.1')
m = svm_train(prob, param)
p_labs, p_acc, p_vals = svm_predict(y_test,x_test,m)
nb_correct = np.sum([p_labs == y_test])
accuracy = nb_correct/nb_test
support_vectors = m.get_SV()
nb_support_vecs = len(support_vectors)
print("SVM Kernel=Linear C=0.1 acc={} n={}".format(accuracy,nb_support_vecs))
# For the format of precomputed kernel, please read LIBSVM README.


##LINEAR KERNEL C=1
prob  = svm_problem(y_train, x_train)
param = svm_parameter('-t 0 -c 1')
m = svm_train(prob, param)
p_labs, p_acc, p_vals = svm_predict(y_test,x_test,m)
nb_correct = np.sum([p_labs == y_test])
accuracy = nb_correct/nb_test
support_vectors = m.get_SV()
nb_support_vecs = len(support_vectors)
print("SVM Kernel=Linear C=1 acc={} n={}".format(accuracy,nb_support_vecs))
# For the format of precomputed kernel, please read LIBSVM README.

##LINEAR KERNEL C=5
prob  = svm_problem(y_train, x_train)
param = svm_parameter('-t 0 -c 5')
m = svm_train(prob, param)
p_labs, p_acc, p_vals = svm_predict(y_test,x_test,m)
nb_correct = np.sum([p_labs == y_test])
accuracy = nb_correct/nb_test
support_vectors = m.get_SV()
nb_support_vecs = len(support_vectors)
print("SVM Kernel=Linear C=5 acc={} n={}".format(accuracy,nb_support_vecs))
# For the format of precomputed kernel, please read LIBSVM README.

##LINEAR KERNEL C=20
prob  = svm_problem(y_train, x_train)
param = svm_parameter('-t 0 -c 20')
m = svm_train(prob, param)
p_labs, p_acc, p_vals = svm_predict(y_test,x_test,m)
nb_correct = np.sum([p_labs == y_test])
accuracy = nb_correct/nb_test
support_vectors = m.get_SV()
nb_support_vecs = len(support_vectors)
print("SVM Kernel=Linear C=20 acc={} n={}".format(accuracy,nb_support_vecs))
# For the format of precomputed kernel, please read LIBSVM README.

##LINEAR KERNEL C=100
prob  = svm_problem(y_train, x_train)
param = svm_parameter('-t 0 -c 100')
m = svm_train(prob, param)
p_labs, p_acc, p_vals = svm_predict(y_test,x_test,m)
nb_correct = np.sum([p_labs == y_test])
accuracy = nb_correct/nb_test
support_vectors = m.get_SV()
nb_support_vecs = len(support_vectors)
print("SVM Kernel=Linear C=100 acc={} n={}".format(accuracy,nb_support_vecs))
# For the format of precomputed kernel, please read LIBSVM README.



##LINEAR KERNEL C=1
prob  = svm_problem(y_train, x_train)
param = svm_parameter('-t 0 -c 1')
m = svm_train(prob, param)
p_labs, p_acc, p_vals = svm_predict(y_test,x_test,m)
nb_correct = np.sum([p_labs == y_test])
accuracy = nb_correct/nb_test
support_vectors = m.get_SV()
nb_support_vecs = len(support_vectors)
print("SVM Kernel=Linear C=1 acc={} n={}".format(accuracy,nb_support_vecs))
# For the format of precomputed kernel, please read LIBSVM README.

# POLYNOMIAL KERNEL C=1
prob  = svm_problem(y_train, x_train)
param = svm_parameter('-t 1 -c 1')
m = svm_train(prob, param)
p_labs, p_acc, p_vals = svm_predict(y_test,x_test,m)
nb_correct = np.sum([p_labs == y_test])
accuracy = nb_correct/nb_test
support_vectors = m.get_SV()
nb_support_vecs = len(support_vectors)
print("SVM Kernel=Polynomial C=1 acc={} n={}".format(accuracy,nb_support_vecs))
# For the format of precomputed kernel, please read LIBSVM README.


##Radial Basis KERNEL C=1
prob  = svm_problem(y_train, x_train)
param = svm_parameter('-t 2 -c 1')
m = svm_train(prob, param)
p_labs, p_acc, p_vals = svm_predict(y_test,x_test,m)
nb_correct = np.sum([p_labs == y_test])
accuracy = nb_correct/nb_test
support_vectors = m.get_SV()
nb_support_vecs = len(support_vectors)
print("SVM Kernel=Radial Basis C=1 acc={} n={}".format(accuracy,nb_support_vecs))
# For the format of precomputed kernel, please read LIBSVM README.


##Sigmoid KERNEL C=1
prob  = svm_problem(y_train, x_train)
param = svm_parameter('-t 3 -c 1')
m = svm_train(prob, param)
p_labs, p_acc, p_vals = svm_predict(y_test,x_test,m)
nb_correct = np.sum([p_labs == y_test])
accuracy = nb_correct/nb_test
support_vectors = m.get_SV()
nb_support_vecs = len(support_vectors)
print("SVM Kernel=Sigmoid C=1 acc={} n={}".format(accuracy,nb_support_vecs))
# For the format of precomputed kernel, please read LIBSVM README.

