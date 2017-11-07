import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn import svm
import random
import warnings as ws


# Ignore all Underflow Warnings : Value close to Zero - No Accuracy Errors
np.seterr(under='ignore')
sp.seterr(under='ignore')
ws.simplefilter('ignore')


def kfold(dataset, k):
    kf = KFold(n_splits=k, shuffle=True, random_state=random.randint(1, 10))
    return kf

# Load the CSV file
dataset = pd.read_csv('voice_mod.csv')
percent = 0.80
dataset_cv = dataset[ : int(percent * dataset.shape[0])]
dataset_main = dataset[int(percent * dataset.shape[0]) : ]

# Get Cross Validation K-Fold
folds = kfold(dataset, 5)

# Get train and test set splits for Cross Validation and Main Testing
for train_index, test_index in folds.split(dataset_cv):
    x_train_cv = np.asarray([[row['IQR'], row['meanfun']] for index, row in dataset_cv.iloc[train_index].iterrows()])
    x_test_cv = np.asarray([[row['IQR'], row['meanfun']] for index, row in dataset_cv.iloc[train_index].iterrows()])
    y_train_cv = np.asarray([[row['label']] for index, row in dataset_cv.iloc[train_index].iterrows()]).ravel()
    y_test_cv = np.asarray([[row['label']] for index, row in dataset_cv.iloc[train_index].iterrows()]).ravel()

x_test_main = np.asarray([[row['IQR'], row['meanfun']] for index, row in dataset_main.iterrows()])
y_test_main = np.asarray([[row['label']] for index, row in dataset_main.iterrows()]).ravel()


#Kernel Hyper Parameters
#Kernel    c    degree    gamma          coef0/r
#         1e3:1e-3  3/poly   rbf/poly/sig   poly/sigmoid 
#linear    above    none      none          none
#poly      above    poly      poly          poly
#rbf/gauss above    none      rbf           none
#sigmoid   above    none      sigmoid       sigmoid


# Tuning for Hyper Parameters

# Linear
t_lin = time.time()
compare_lin = []
print ('Linear Kernel')
for c_lin in [10.0, 50.0, 100.0, 250.0, 500.0, 750.0, 1000.0]:
    #print ('c ' + str(c))
    svc = svm.SVC(kernel='linear', C=c_lin, cache_size=8192)
    t = time.time()
    svc.fit(x_train_cv, y_train_cv)
    svc_fit = time.time() - t
    #print('Time Taken to Fit the Model : ' + str(svc_fit) + 'Secs')
    t = time.time()
    y_svc = svc.predict(x_test_cv)
    svc_predict = time.time() - t
    #print('Time Taken to Predict using the Fitted Model : ' + str(svc_predict) + 'Secs')
    acc_score = r2_score(y_test_cv, y_svc) * 100
    #print('Accuracy Score of the Model : ' + str(acc_score))
    compare_lin.append([c_lin, svc_fit, svc_predict, acc_score])
t_lin = time.time() - t_lin
print('Time Taken to Tune Linear Kernel Hyper-Parameter Values = ' + str(t_lin) + ' Secs \n')

# Gaussian
t_rbf = time.time()
compare_rbf =[]
print ('Gaussian Kernel')
for c_rbf in [10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0]:
    #print ('c ' + str(c))
    for g_rbf in [1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0]:
        #print ('s ' + str(s))
        svc = svm.SVC(kernel='rbf', C=c_rbf, gamma=g_rbf, cache_size=8192)
        t = time.time()
        svc.fit(x_train_cv, y_train_cv)
        svc_fit = time.time() - t
        #print('Time Taken to Fit the Model : ' + str(svc_fit) + 'Secs')
        t = time.time()
        y_svc = svc.predict(x_test_cv)
        svc_predict = time.time() - t
        #print('Time Taken to Predict using the Fitted Model : ' + str(svc_predict) + 'Secs')
        acc_score = r2_score(y_test_cv, y_svc) * 100
        #print('Accuracy Score of the Model : ' + str(acc_score))
        compare_rbf.append([c_rbf, g_rbf, svc_fit, svc_predict, acc_score])
t_rbf = time.time() - t_rbf
print('Time Taken to Tune Gaussian Kernel Hyper-Parameter Values = ' + str(t_rbf) + ' Secs \n')

# Sigmoid
t_sgm = time.time()
compare_sigmoid =[]
print ('Sigmoid Kernel')
for c_sgm in [10.0, 50.0, 100.0, 500.0, 1000.0]:
    #print ('c ' + str(c))
    for g_sgm in [2.5, 3.0, 3.5]:
        #print ('g ' + str(g))
        for r_sgm in [1.5, 2.0, 2.5]:
            #print ('r ' + str(r))
            svc = svm.SVC(kernel='sigmoid', C=c_sgm, gamma=g_sgm, coef0=r_sgm, cache_size=8192)
            t = time.time()
            svc.fit(x_train_cv, y_train_cv)
            svc_fit = time.time() - t
            #print('Time Taken to Fit the Model : ' + str(svc_fit) + 'Secs')
            t = time.time()
            y_svc = svc.predict(x_test_cv)
            svc_predict = time.time() - t
            #print('Time Taken to Predict using the Fitted Model : ' + str(svc_predict) + 'Secs')
            acc_score = r2_score(y_test_cv, y_svc) * 100
            #print('Accuracy Score of the Model : ' + str(acc_score))
            compare_sigmoid.append([c_sgm, g_sgm, r_sgm, svc_fit, svc_predict, acc_score])
t_sgm = time.time() - t_sgm
print('Time Taken to Tune Sigmoid Kernel Hyper-Parameter Values = ' + str(t_sgm) + ' Secs \n')

# Polynomial
t_poly = time.time()
compare_poly = []
print ('Polynomial Kernel')
for c_poly in [10.0, 50.0, 100.0, 500.0, 1000.0]:
    #print ('c ' + str(c))
    for d_poly in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]:
        #print ('d ' + str(d))
        for g_poly in [1.0, 5.0, 10.0, 50.0]:
            #print ('g ' + str(g))
            for r_poly in [0.5, 1.0, 1.5]:
                #print ('r ' + str(r))
                svc = svm.SVC(kernel='poly', C=c_poly, degree=d_poly, gamma=g_poly, coef0=r_poly, cache_size=8192)
                t = time.time()
                svc.fit(x_train_cv, y_train_cv)
                svc_fit = time.time() - t
                #print('Time Taken to Fit the Model : ' + str(svc_fit) + 'Secs')
                t = time.time()
                y_svc = svc.predict(x_test_cv)
                svc_predict = time.time() - t
                #print('Time Taken to Predict using the Fitted Model : ' + str(svc_predict) + 'Secs')
                acc_score = r2_score(y_test_cv, y_svc) * 100
                #print('Accuracy Score of the Model : ' + str(acc_score))
                compare_poly.append([c_poly, d_poly, g_poly, r_poly, svc_fit, svc_predict, acc_score])
t_poly = time.time() - t_poly
print('Time Taken to Tune Polynomial Kernel Hyper-Parameter Values = ' + str(t_poly) + ' Secs \n')

# Total Time Taken
print('Total Time Taken to Tune Hyper-Parameter Values = ' + str((t_lin + t_rbf + t_sgm + t_poly) / 60) + ' Mins \n')


# Saving All Values obtained from Hyper Parameter Tuning
compare_lin = np.array(compare_lin)
df = pd.DataFrame(compare_lin)
df.to_csv("linear_svm_results_sklearn.csv")
compare_rbf = np.array(compare_rbf)
df = pd.DataFrame(compare_rbf)
df.to_csv("gaussian_svm_results_sklearn.csv")
compare_sigmoid = np.array(compare_sigmoid)
df = pd.DataFrame(compare_sigmoid)
df.to_csv("sigmoid_svm_results_sklearn.csv")
compare_poly = np.array(compare_poly)
df = pd.DataFrame(compare_poly)
df.to_csv("polynomial_svm_results_sklearn.csv")


# Fetching Hyperparameters based on Max Accuracy Score and Return Kernel

# Linear
lin_max = np.where(compare_lin[:, 3] == np.amax(compare_lin[:, 3]))
lin_c = compare_lin[lin_max[0]][0][0]

# RBF/Gaussian
rbf_max = np.where(compare_rbf[:, 4] == np.amax(compare_rbf[:, 4]))
rbf_c = compare_rbf[rbf_max[0]][0][0]
rbf_sigma = compare_rbf[rbf_max[0]][0][1]

# Sigmoid
sigmoid_max = np.where(compare_sigmoid[:, 5] == np.amax(compare_sigmoid[:, 5]))
sigmoid_c = compare_sigmoid[sigmoid_max[0]][0][0]
sigmoid_gamma = compare_sigmoid[sigmoid_max[0]][0][1]
sigmoid_coef = compare_sigmoid[sigmoid_max[0]][0][2]

# Polynomial
poly_max = np.where(compare_poly[:, 6] == np.amax(compare_poly[:, 6]))
poly_c = compare_poly[poly_max[0]][0][0]
poly_deg = compare_poly[poly_max[0]][0][1]
poly_gamma = compare_poly[poly_max[0]][0][2]
poly_coef = compare_poly[poly_max[0]][0][3]

# Saving the Best Values obtained from Hyper Parameter Tuning
best_hyparams = np.array([[lin_c, None, None, None], 
                          [rbf_c, None, rbf_sigma, None], 
                          [sigmoid_c, None, sigmoid_gamma, sigmoid_coef], 
                          [poly_c, poly_deg, poly_gamma, poly_coef]], dtype=float)
best_hyparams = best_hyparams.reshape(4, 4)
df = pd.DataFrame(best_hyparams)
df.to_csv("hyparams_svm_results_sklearn.csv")

# Generate Prediction for selected Hyperparameters

#Linear
print('\nLinear Kernel with Hyperparameters : C = ' + str(lin_c))
svc_lin = svm.SVC(kernel='linear', C=lin_c, cache_size=4096)
t = time.time()
fit_lin = svc_lin.fit(x_test_main, y_test_main)
svc_fit = time.time() - t
print('Time Taken to Fit the Model : ' + str(svc_fit) + ' Secs')
t = time.time()
y_lin = svc_lin.predict(x_test_main)
svc_predict = time.time() - t
print('Time Taken to Predict using the Fitted Model : ' + str(svc_predict) + ' Secs')
acc_lin = r2_score(y_test_main, y_lin) * 100
print('Accuracy Score of the Model : ' + str(acc_lin))
plt.figure('Linear Kernel SVM ~ SKLearn')
plt.plot([x for x in range(0, len(y_test_main))], y_test_main, 'b.')
plt.plot([x for x in range(0, len(y_test_main))], y_lin, 'r.')
plt.title('Linear Kernel SVM ~ Y_test vs. Y_pred')
plt.legend(['Y_test', 'Y_pred'])

#Gaussian
print('\nGaussian Kernel with Hyperparameters : C = ' + str(rbf_c) + 
      ', Sigma = ' + str(rbf_sigma))
svc_rbf = svm.SVC(kernel='rbf', C=rbf_c, gamma=rbf_sigma, cache_size=4096)
t = time.time()
fit_rbf = svc_rbf.fit(x_test_main, y_test_main)
svc_fit = time.time() - t
print('Time Taken to Fit the Model : ' + str(svc_fit) + ' Secs')
t = time.time()
y_rbf = svc_rbf.predict(x_test_main)
svc_predict = time.time() - t
print('Time Taken to Predict using the Fitted Model : ' + str(svc_predict) + ' Secs')
acc_rbf = r2_score(y_test_main, y_rbf) * 100
print('Accuracy Score of the Model : ' + str(acc_rbf))
plt.figure('Gaussian Kernel SVM ~ SKLearn')
plt.plot([x for x in range(0, len(y_test_main))], y_test_main, 'b.')
plt.plot([x for x in range(0, len(y_test_main))], y_rbf, 'r.')
plt.title('Gaussian Kernel SVM ~ Y_test vs. Y_pred')
plt.legend(['Y_test', 'Y_pred'])

#Sigmoid
print('\nSigmoid Kernel with Hyperparameters : C = ' + str(sigmoid_c) + 
      ', Gamma = ' + str(sigmoid_gamma) + ', Coeff = ' + str(sigmoid_coef))
svc_sgm = svm.SVC(kernel='sigmoid', C=sigmoid_c, gamma=sigmoid_gamma, coef0=sigmoid_coef, cache_size=4096)
t = time.time()
fit_sgm = svc_sgm.fit(x_test_main, y_test_main)
svc_fit = time.time() - t
print('Time Taken to Fit the Model : ' + str(svc_fit) + ' Secs')
t = time.time()
y_sgm = svc_sgm.predict(x_test_main)
svc_predict = time.time() - t
print('Time Taken to Predict using the Fitted Model : ' + str(svc_predict) + ' Secs')
acc_sgm = r2_score(y_test_main, y_sgm) * 100
print('Accuracy Score of the Model : ' + str(acc_sgm))
plt.figure('Sigmoid Kernel SVM ~ SKLearn')
plt.plot([x for x in range(0, len(y_test_main))], y_test_main, 'b.')
plt.plot([x for x in range(0, len(y_test_main))], y_sgm, 'r.')
plt.title('Sigmoid Kernel SVM ~ Y_test vs. Y_pred')
plt.legend(['Y_test', 'Y_pred'])

#Polynomial
print('\nPolynomial Kernel with Hyperparameters : C = ' + str(poly_c) + 
      ', Degree = ' + str(poly_deg) + ', Gamma = ' + str(poly_gamma) + 
      ', Coeff = ' + str(poly_coef))
svc_poly = svm.SVC(kernel='poly', C=poly_c, degree=poly_deg, gamma=poly_gamma, coef0=poly_coef, cache_size=4096)
t = time.time()
fit_poly = svc_poly.fit(x_test_main, y_test_main)
svc_fit = time.time() - t
print('Time Taken to Fit the Model : ' + str(svc_fit) + ' Secs')
t = time.time()
y_poly = svc_poly.predict(x_test_main)
svc_predict = time.time() - t
print('Time Taken to Predict using the Fitted Model : ' + str(svc_predict) + ' Secs')
acc_poly = r2_score(y_test_main, y_poly) * 100
print('Accuracy Score of the Model : ' + str(acc_poly))
plt.figure('Polynomial Kernel SVM ~ SKLearn')
plt.plot([x for x in range(0, len(y_test_main))], y_test_main, 'b.')
plt.plot([x for x in range(0, len(y_test_main))], y_poly, 'r.')
plt.title('Polynomial Kernel SVM ~ Y_test vs. Y_pred')
plt.legend(['Y_test', 'Y_pred'])

# Contour Graphs
def make_meshgrid(x1, x2, h=0.01):
    # Create a mesh of points to plot
    x1_min, x1_max = x1.min() - 1, x1.max() + 1
    x2_min, x2_max = x2.min() - 1, x2.max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
    return xx1, xx2

def plot_contours(ax, clf, xx1, xx2, **params):
    # Plot the decision boundaries for a classifier.
    Z = clf.predict(np.c_[xx1.ravel(), xx2.ravel()])
    Z = Z.reshape(xx1.shape)
    out = ax.contour(xx1, xx2, Z, **params)
    return out

model_fits = (fit_lin, fit_rbf, fit_sgm, fit_poly)
titles = ('Linear Kernel SVM', 'Gaussian Kernel SVM', 'Sigmoidal Kernel SVM', 'Polynomial Kernel SVM')
figure, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.5, hspace=0.5)

x1 = np.asarray([[row['IQR']] for index, row in dataset_main.iterrows()]).ravel()
x2 = np.asarray([[row['meanfun']] for index, row in dataset_main.iterrows()]).ravel()
yy = np.asarray([[row['label']] for index, row in dataset_main.iterrows()]).ravel()
xx1, xx2 = make_meshgrid(x1, x2)

figure.canvas.set_window_title('Scatter Contour SVM Plot ~ SKLearn')
for fit, title, fig in zip(model_fits, titles, sub.flatten()):
    plot_contours(fig, fit, xx1, xx2, cmap=plt.cm.bwr, alpha=0.8)
    fig.scatter(x1, x2, c=yy, cmap=plt.cm.bwr, s=20, edgecolors='k')
    fig.set_xlim(x1.min()-(np.mean(x1)/4), x1.max()+(np.mean(x1)/4))
    fig.set_ylim(x2.min()-(np.mean(x2)/4), x2.max()+(np.mean(x2)/4))
    fig.set_xlabel('IQR [Blue]')
    fig.set_ylabel('MeanFun [Red]')
    fig.set_xticks(())
    fig.set_yticks(())
    fig.set_title(title)

# Reset Underflow Warnings : Value close to Zero - No Accuracy Errors
np.seterr(under='warn')
sp.seterr(under='warn')
ws.resetwarnings()


# End of File