# -*- coding: cp1252 -*-
# Kimberly Kaminsky - Assignment #3
# Exploring MNIST with Multi-Class Classification 

# Data from MNIST may be used to evaluate machine learning classifiers.
# Here we will use a subset of the MNIST data to study binary classifiers.
# In particular, after exploring the entire MNIST data set, we will 
# select a subset of the data... just the digits 0 and 6

# to obtain a listing of the results of this program, 
# locate yourself in the working direcotry and
# execute the following command in a terminal or commands window
# python exploring-mnist-v001.py > listing-exploring-mnist-v001.txt

#################### 
# Define constants #
####################

# seed value for random number generators to obtain reproducible results
RANDOM_SEED = 11
RANDOM_SEED_MODEL = 111

# number of folds for cross validation
N_FOLDS = 10


####################
# Import Libraries #
####################

# Import base packages into the namespace for this program
import numpy as np
import pandas as pd
import os
import time
import matplotlib as mpl
import warnings
import PyPDF2 as pp                      # Allows pdf file manipulation

# Visualization utilities
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages  # plot to pdf files

# Cross-validation scoring code adapted from Scikit Learn documentation
from sklearn.metrics import f1_score

# Modeling routines
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV 
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

# Import data from Scikit Learn
from sklearn.datasets import fetch_mldata

# Provides random sampling
from sklearn.utils import resample

#############
# Functions #
#############

# user-defined function for displaying observations/handwritten digits
# adapted from Géron (2017) Python notebook code (default 10 images per row)
def plot_digits(instances, images_per_row = 10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = matplotlib.cm.binary, **options)
    plt.axis('off')
    
# function to clear output console
def clear():
    print("\033[H\033[J")
    
# --------------------------------------------------------


####################
# Data Exploration #
####################

# Set current datapath
datapath = os.path.join("D:/","Kim MSPA", "Predict 422", "Assignments", "Assignment5", "")

# Turn off interactive mode since a large number of plots are generated
# Plots will be saved off in pdf files
mpl.is_interactive()
plt.ioff()

# Explore the data
mnist = fetch_mldata('MNIST original')
mnist  # show structure of datasets Bunch object from Scikit Learn

# define arrays from the complete data set
mnist_X, mnist_y = mnist['data'], mnist['target']

# show stucture of numpy arrays
# 70,000 observations, 784 explanatory variables/features
# features come from 28x28 pixel displays
# response is a single digit 0 through 9
print('\n Structure of explanatory variable array:', mnist_X.shape)
print('\n Structure of response array:', mnist_y.shape)

# route plot to external pdf file
with PdfPages(datapath + 'plot-mnist-index-plot.pdf') as pdf:
    fig, axis = plt.subplots()
    axis.set_xlabel('Sequence/Index number within MNIST Data Set')
    axis.set_ylabel('MNIST Digit')
    plt.title('Index Plot of MNIST Data Set')
    plt.plot(mnist_y[:,])
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()

# summarize the sequential structure of the MNIST data
# target/label and index values for the observations  
# examine the frequency distributions for the digits using pandas DataFrame
# the first 60 thousand observations are used as training data    
mnist_y_0_59999_df = pd.DataFrame({'label': mnist_y[0:59999,]}) 
print('\nFrequency distribution for 60,000 observations (for model building)')
print(mnist_y_0_59999_df['label'].value_counts(ascending = True))   

# the last 10000 observations cover the ten digits
# these are often used as test data
# digits are arranged in order but the frequencies are unequal     
mnist_y_60000_69999_df = pd.DataFrame({'label': mnist_y[60000:69999,]}) 
print('\nFrequency distribution for last 10,000 observations (holdout sample)')
print(mnist_y_60000_69999_df['label'].value_counts(ascending = True))   

# display example data from the 28x28 pixel displays 
# ten-page pdf file, 100 digit realizations on each page
# using examples from the full MNIST data set
# begin by showing samples from the model building data (first 60000 observations)  
with PdfPages(datapath + 'plot-mnist-handwritten-digits-model-building-data.pdf') as pdf:
    for idigit in range(0,10):
        # print('\nworking on digit', idigit)
        
        # identify the index values from the first 60000 observations
        # that have the label equal to a specific digit (idigit)
        idigit_indices = \
            mnist_y_0_59999_df.index[mnist_y_0_59999_df.label == idigit]
        # obtain indices for 100 randomly sampled observations for this digit    
        show_indices = resample(idigit_indices, n_samples=100, 
                                replace = False, 
                                random_state = RANDOM_SEED).sort_values()       
        plt.figure(0)
        plt.suptitle('Examples of MNIST Data for Digit ' + str(idigit))
        # define beginning and ending row index for this digit
        # generate ten rows of ten digits each
        for j in range(0,10):
            row_begin_index = j * 10
            row_end_index = row_begin_index + 10
            # print('row begin',row_begin_index, 'row_end', row_end_index)
            this_row_indices = show_indices[row_begin_index:row_end_index]
            
            example_images = np.r_[mnist_X[this_row_indices]]
            # print(mnist_y[this_row_indices,])
            plt.subplot2grid((10,1), (j,0), colspan=1)
            # plot ten digits per row using user-defined function
            plot_digits(example_images, images_per_row=10)
            row_begin_index = row_end_index + 1
        pdf.savefig()  
        plt.close()   

# also show samples from the holdout data (last 10000 observations)  
with PdfPages(datapath + 'plot-mnist-handwritten-digits-holdout-data.pdf') as pdf:
    for idigit in range(0,10):
        # print('\nworking on digit', idigit)
        
        # identify the index values from the first 60000 observations
        # that have the label equal to a specific digit (idigit)
        idigit_indices = 60000 + \
        mnist_y_60000_69999_df.index[mnist_y_60000_69999_df.label == idigit]
        # obtain indices for 100 randomly sampled observations for this digit    
        show_indices = resample(idigit_indices, n_samples=100, 
                                replace = False, 
                                random_state = RANDOM_SEED).sort_values()       
        plt.figure(0)
        plt.suptitle('Examples of MNIST Data for Digit ' + str(idigit))
        # define beginning and ending row index for this digit
        # generate ten rows of ten digits each
        for j in range(0,10):
            row_begin_index = j * 10
            row_end_index = row_begin_index + 10
            # print('row begin',row_begin_index, 'row_end', row_end_index)
            this_row_indices = show_indices[row_begin_index:row_end_index]
            
            example_images = np.r_[mnist_X[this_row_indices]]
            # print(mnist_y[this_row_indices,])
            plt.subplot2grid((10,1), (j,0), colspan=1)
            # plot ten digits per row using user-defined function
            plot_digits(example_images, images_per_row=10)
            row_begin_index = row_end_index + 1
        pdf.savefig()  
        plt.close()   


with PdfPages(datapath + 'plot-one-page-all-digits.pdf') as pdf:
    fig = plt.figure(figsize=(9,9))
    example_images = np.r_[mnist_X[:12000:600], mnist_X[13000:30600:600], mnist_X[30600:60000:590]]
    plot_digits(example_images, images_per_row=10)
    #save_fig("more_digits_plot")
    pdf.savefig(fig)
        
        
###########################
# Models and Benchmarking #
###########################   

# Turn off future warnings as adding return_train_score parameter 
# doesn't work and it gives me a future warning after every print command
warnings.simplefilter(action='ignore', category=FutureWarning)
               

# Training data
model_y = np.r_[mnist_y_0_59999_df]
model_X = np.r_[mnist_X[0:59999,]]
model_data = np.concatenate((model_y.reshape(-1, 1), model_X), axis = 1)

# check on shape of the model_data array
print('\nShape of model_data:', model_data.shape)        

# Test data
holdout_y = np.r_[mnist_y_60000_69999_df]
holdout_X = np.r_[mnist_X[60000:69999,]]
holdout_data = np.concatenate((holdout_y.reshape(-1, 1), 
                               holdout_X), axis = 1)

# check on shape of the holdout_data array
print('\nShape of holdout_data:', holdout_data.shape)


# shuffle the rows because MNIST data rows have a sequence
# with lower digits coming before higher digits
# shuffle is by the first index, which is the rows
np.random.seed(RANDOM_SEED)
np.random.shuffle(model_data)

np.random.seed(RANDOM_SEED)
np.random.shuffle(holdout_data)

# Setup model to run
names = ["Random Forest"]
classifiers = [RandomForestClassifier(n_estimators=10, max_features='sqrt', 
                            bootstrap=True)]

# set up numpy array for storing results
cv_results = np.zeros((N_FOLDS, len(names)))

# Run Random Forest Classifier using 10 fold corss validation with timer
start_time = time.clock()

kf = KFold(n_splits = N_FOLDS, shuffle=False, random_state = RANDOM_SEED)
# check the splitting process by looking at fold observation counts
index_for_fold = 0  # fold count initialized 
for train_index, test_index in kf.split(model_data):
    print('\nFold index:', index_for_fold,
          '------------------------------------------')
#   the structure of modeling data for this study has the
#   response variable coming first and explanatory variables later          
#   so 1:model_data.shape[1] slices for explanatory variables
#   and 0 is the index for the response variable    
    X_train = model_data[train_index, 1:model_data.shape[1]]
    X_test = model_data[test_index, 1:model_data.shape[1]]
    y_train = model_data[train_index, 0]
    y_test = model_data[test_index, 0]   
    print('\nShape of input data for this fold:',
          '\nData Set: (Observations, Variables)')
    print('X_train:', X_train.shape)
    print('X_test:',X_test.shape)
    print('y_train:', y_train.shape)
    print('y_test:',y_test.shape)

    index_for_method = 0  # initialize
    for name, clf in zip(names, classifiers):
        print('\nClassifier evaluation for:', name)
        print('  Scikit Learn method:', clf)
        clf.fit(X_train, y_train)  # fit on the train set for this fold
        # evaluate on the test set for this fold
        y_test_predict = clf.predict(X_test)
        fold_method_result = f1_score(y_test,y_test_predict,average='weighted') 
        print('F1 Score:', fold_method_result)
        cv_results[index_for_fold, index_for_method] = fold_method_result
        index_for_method += 1
  
    index_for_fold += 1

cv_results_df = pd.DataFrame(cv_results)
cv_results_df.columns = names

print('\n----------------------------------------------')
print('Average results from ', N_FOLDS, '-fold cross-validation\n',
      '\nMethod                 F1 Score', sep = '')     
print(cv_results_df.mean())   

end_time = time.clock()
runtime = end_time - start_time    
 
print('\nRuntime for Random Forest:', runtime)       
             
             
# Run pricipal component analysis with timer                  
start_time_pca = time.clock()  

pca = PCA(n_components=0.95)
reduced = pca.fit_transform(mnist_X)
                     
end_time_pca = time.clock()
runtime_pca = end_time_pca - start_time_pca    

runtime_pca    


# Create train and test set on reduced data
train = np.concatenate((model_y.reshape(-1, 1), reduced[0:59999,]), axis = 1)
test = np.concatenate((holdout_y.reshape(-1, 1), 
                               reduced[60000:69999,]), axis = 1)
                               
# shuffle the rows because MNIST data rows have a sequence
# with lower digits coming before higher digits
# shuffle is by the first index, which is the rows
np.random.seed(RANDOM_SEED)
np.random.shuffle(train)

np.random.seed(RANDOM_SEED)
np.random.shuffle(test)


print('\nShape of train:', train.shape)    
print('\nShape of test:', test.shape)    


# Run random forest on reduced dataset
cv_resultsPCA = np.zeros((N_FOLDS, len(names)))

start_time_reducedRF = time.clock()

kf = KFold(n_splits = N_FOLDS, shuffle=False, random_state = RANDOM_SEED)
# check the splitting process by looking at fold observation counts
index_for_fold = 0  # fold count initialized 
for train_index, test_index in kf.split(train):
    print('\nFold index:', index_for_fold,
          '------------------------------------------')
#   the structure of modeling data for this study has the
#   response variable coming first and explanatory variables later          
#   so 1:model_data.shape[1] slices for explanatory variables
#   and 0 is the index for the response variable    
    X_train = train[train_index, 1:train.shape[1]]
    X_test = train[test_index, 1:train.shape[1]]
    y_train = train[train_index, 0]
    y_test = train[test_index, 0]   
    print('\nShape of input data for this fold:',
          '\nData Set: (Observations, Variables)')
    print('X_train:', X_train.shape)
    print('X_test:',X_test.shape)
    print('y_train:', y_train.shape)
    print('y_test:',y_test.shape)

    index_for_method = 0  # initialize
    for name, clf in zip(names, classifiers):
        print('\nClassifier evaluation for:', name)
        print('  Scikit Learn method:', clf)
        clf.fit(X_train, y_train)  # fit on the train set for this fold
        # evaluate on the test set for this fold
        y_test_predict = clf.predict(X_test)
        fold_method_result = f1_score(y_test,y_test_predict,average='weighted') 
        print('F1 Score:', fold_method_result)
        cv_resultsPCA[index_for_fold, index_for_method] = fold_method_result
        index_for_method += 1
  
    index_for_fold += 1

cv_resultsPCA_df = pd.DataFrame(cv_resultsPCA)
cv_resultsPCA_df.columns = names

print('\n----------------------------------------------')
print('Average results from ', N_FOLDS, '-fold cross-validation\n',
      '\nMethod                 F1 Score', sep = '')     
print(cv_resultsPCA_df.mean())   

end_time_reducedRF = time.clock()
runtime_reducedRF = end_time_reducedRF - start_time_reducedRF  

# Total time for PCA-based model
runtime_total_pca = runtime_pca + runtime_reducedRF
                       
print('\nRuntime for Dimension Reduced Random Forest:', runtime_total_pca)   

################
# Score Models # 
################

# Final run of the models on holdout data
# Pure Random Forest Classifier
rnd_clf = RandomForestClassifier(n_estimators=10, max_features='sqrt', 
                             bootstrap=True)
rnd_clf.fit(model_data[:,1:model_data.shape[1]], model_data[:,0])
y_predrf = rnd_clf.predict(holdout_X)

# Dimension reduced Random Forest Classifier

rrnd_clf = RandomForestClassifier(n_estimators=10, max_features='sqrt', 
                            bootstrap=True)
rrnd_clf.fit(train[:,1:train.shape[1]], train[:,0])
y_predrrf = rrnd_clf.predict(test[:,1:test.shape[1]])


pure_rf = f1_score(y_predrf, holdout_y, average='weighted')

pure_rrf = f1_score(y_predrrf, test[:,0], average='weighted')

###########
# Summary #
###########

print('\nRuntime for Random Forest:', runtime)   
print('\nRuntime for Dimension Reduced Random Forest:', runtime_total_pca)   

print('\nPure Random Forest:', pure_rf)    
print('\nReduced Dimension Random Forest:', pure_rrf)                                                                           
                                                                                  
                                                                                  
                                                                                  
#########################################
# Combine all pdf files into 1 pdf file #
#########################################

pdfWriter = pp.PdfFileWriter()
pdfOne = pp.PdfFileReader(open(datapath + "plot-one-page-all-digits.pdf", "rb"))
pdfTwo = pp.PdfFileReader(open(datapath + "plot-mnist-index-plot.pdf", "rb"))
pdfThree = pp.PdfFileReader(open(datapath + "plot-mnist-handwritten-digits-holdout-data.pdf", "rb"))
pdfFour = pp.PdfFileReader(open(datapath + "plot-mnist-handwritten-digits-model-building-data.pdf", "rb"))

for pageNum in range(pdfOne.numPages):        
    pageObj = pdfOne.getPage(pageNum)
    pdfWriter.addPage(pageObj)


for pageNum in range(pdfTwo.numPages):        
    pageObj = pdfTwo.getPage(pageNum)
    pdfWriter.addPage(pageObj)
    
for pageNum in range(pdfThree.numPages):        
    pageObj = pdfThree.getPage(pageNum)
    pdfWriter.addPage(pageObj)

for pageNum in range(pdfFour.numPages):        
    pageObj = pdfFour.getPage(pageNum)
    pdfWriter.addPage(pageObj)

outputStream = open(datapath + r"Assign5_Output.pdf", "wb")
pdfWriter.write(outputStream)
outputStream.close()

                                
                                


                                                            