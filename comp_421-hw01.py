# coding: utf-8
import csv 
import numpy as np
import pandas as pd

# Reading the image data into a list of lists.
with open("hw01_images.csv", 'r') as f:
    reader = csv.reader(f)
    data = list(list(cont) for cont in csv.reader(f, delimiter=','))
    f.close()
    
# Reading the labels into a list.
labels = list(csv.reader(open("hw01_labels.csv", "r"), delimiter=","))

# Splitting the read data into two arrays, namely test_data and train_data.
train_data = data[:len(data)//2]
test_data = data[len(data)//2:]  

# Saving the labels as keys in a map, and separating the training data depending on the corresponding labels.
sep_data = {}
for i in range(len(labels[0:200])):
    if (labels[i][0] not in sep_data):
        sep_data[labels[i][0]] = list()    
    sep_data[labels[i][0]].append(train_data[i])
# Extracting unique labels into an array.
unique_labels = np.unique(labels)

# Calculating means
means = {}
for i in range(len(unique_labels)):
    means[unique_labels[i]] = list()
    rows, cols = np.shape(sep_data[unique_labels[i]])
    for j in range(cols):
        cur_data = [float(x) for x in [row[j] for row in sep_data[unique_labels[i]]]]
        means[unique_labels[i]].append(sum(cur_data)/len(cur_data))
# Printing the mean values for corresponding labels.
# for i in range(len(unique_labels)):
#    print("means {0}: {1}".format(unique_labels[i],means[unique_labels[i]]))

# Calculating standard deviations
std_devs = {}
for i in range(len(unique_labels)):
    std_devs[unique_labels[i]] = list()
    rows, cols = np.shape(sep_data[unique_labels[i]])
    for j in range(cols):
        mean = means[unique_labels[i]][j]
        cur_data = [float(x) for x in [row[j] for row in sep_data[unique_labels[i]]]]
        var = sum([pow(d-mean,2) for d in cur_data])/(len(cur_data)-1)
        std_devs[unique_labels[i]].append(np.sqrt(var))
# Printing the standard deviation values for corresponding labels.
# for i in range(len(unique_labels)):
#    print("standard deviations {0}: {1}\n".format(unique_labels[i],std_devs[unique_labels[i]]))

# Calculating prior probs
priors = {}
for i in range(len(unique_labels)):
    priors[unique_labels[i]] = len(sep_data[unique_labels[i]])/len(train_data)
# Printing the prior probabilities.
# for i in range(len(unique_labels)):
#    print("prior {0}: {1}\n".format(unique_labels[i],priors[unique_labels[i]]))

# Gaussian probability density function, since the inputs are continuous.
def likelihood(data, mu, sigma):
    return np.exp(-(data-mu)**2/(2*sigma**2)) *  (1 / np.sqrt(2*np.pi)*sigma)
# Defining the safelog function.
def safelog(x):
    return np.log(x + 1e-100)

# Calculating the log_likelihoods for both the train and test data sets.
log_likelihoods_train = {}
log_likelihoods_test = {}
for i in range(len(unique_labels)):
    log_likelihoods_train[unique_labels[i]] = list()
    log_likelihoods_test[unique_labels[i]] = list()
    rows, cols = np.shape(train_data) # assuming the training the test data sets are split equally (for this hw).
    for j in range(rows):
        cur_data_train = [float(x) for x in train_data[j]]
        cur_data_test = [float(x) for x in test_data[j]]
        log_likelihood_train = 0
        log_likelihood_test = 0
        for k in range(cols):
            mean = means[unique_labels[i]][k]
            std_dev = std_devs[unique_labels[i]][k]
            log_likelihood_train += safelog(likelihood(cur_data_train[k],mean,std_dev))
            log_likelihood_test += safelog(likelihood(cur_data_test[k],mean,std_dev))
        log_likelihoods_train[unique_labels[i]].append(log_likelihood_train) 
        log_likelihoods_test[unique_labels[i]].append(log_likelihood_test) 

# Calculating score values for both the train and test data sets.
log_posteriors_train = {}
log_posteriors_test = {}
for i in range(len(unique_labels)):
    log_posteriors_train[unique_labels[i]] = list()
    log_posteriors_test[unique_labels[i]] = list()
    for j in range(len(log_likelihoods_train[unique_labels[i]])):
        log_posteriors_train[unique_labels[i]].append(log_likelihoods_train[unique_labels[i]][j] + safelog(priors[unique_labels[i]]))
        log_posteriors_test[unique_labels[i]].append(log_likelihoods_test[unique_labels[i]][j] + safelog(priors[unique_labels[i]]))

"""
Configuring the prediction data, depending on the resulting posterior probabilities. Placing either '1' or '2' label
depending on the comparison of posterior probabilities, on the corresponding index of each data.
"""
y_train = list()
y_test = list()
for i in range(len(train_data)):
    y_train.append(2 if log_posteriors_train['2'][i] > log_posteriors_train['1'][i] else 1)
    y_test.append(2 if log_posteriors_test['2'][i] > log_posteriors_test['1'][i] else 1)
# Just restoring the labels as integers.
y_hat = list()
for i in range (int(len(labels))):
    y_hat.append(int(labels[i][0]))

# Confusion matrix for training set.
confusion_matrix_tr = pd.crosstab(np.array(y_hat[:len(y_hat)//2]), np.array(y_train), rownames = ['y_hat'], colnames = ['y_train'])
print(confusion_matrix_tr.head())

# Confusion matrix for test set.
confusion_matrix_te = pd.crosstab(np.array(y_hat[len(y_hat)//2:]), np.array(y_test), rownames = ['y_hat'], colnames = ['y_test'])
print(confusion_matrix_te.head())
