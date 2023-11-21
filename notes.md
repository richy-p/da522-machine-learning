# DASC 522 Machine Learning
---
# Week 1
# Readings
* DoD AI StrategyDownload DoD AI Strategy
* Executive Order on AIDownload Executive Order on AI
* IBM CRISP-DM GuideDownload IBM CRISP-DM Guide
* "ISLP" Chapter 1 & 2
    * Introduction to Statistical Learning with Applications in Python. James, Witten, Hastie, and Tibshirani 
* "HOML" Chapter 1
    * Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems 3rd Edition. Aurelien Geron
    * In this text, please avoid "pipeline" examples - machine learning pipelines are useful for more complex implementations, and aren't required in this course

# 1A Course Introduction
* Machine learning taxonomy
* Secretary of Defense AI motivation
* Instructor introduction
* Course objectives
* Link between certificate courses

# 1B Intro to Machine Learning
* Machine Learning Overview
* DoD Joint Artificial Intelligence Center (JAIC) Overview
* What you can & can’t do with Machine Learning
* Quality of Fit

## Machine Learning Overview
* Taxonomy chart
### Types of ML
* Supervised Learning
* Unsupervised Learning
* Reinforcement Learning

## What you can & can't do with ML
* Parametric vs Non-parametric Methods
* Flexibility

## Quality of Fit
* mean squared error (MSE)
* Training and Test sets performance




# 1C Decision Support & DoD Machine Learning
* Decision Support Motivation
* DoD & USAF AI/ML Strategic Guidance

## 



# 1D Intro to Colab
* Accessing Colab
* Code markdown
* Importing and working with excel data

---
# Week 2

# Regression review (iris example)
* AI Motivation from the Vice Chairman of the Joint Chiefs of Staff
* Regression Fundamentals
* Measuring Error
* Python Model Building and Troubleshooting


# Classification Part 1 and Part 2 
* Logistic Regression
* Linear Discriminant Analysis
* Quadratic Discriminant Analysis
* Classification Thresholds and Measuring Error
* Note: in the Part 2 video the COVID material is from this linkLinks to an external site. 
* Binary Classification metric summary
* Python Model Building: Python (v3) file Download Python (v3) file and GRE data file Download GRE data file 


# Tree-based regression and classification 
* Regression & Classification Trees
* Regularization via Pruning
* Trees vs. Linear Models
* Advantages and Disadvantages of Trees


---
# Week 3
# Clustering overview and k-means fundamentals
* Python Scikit-Learn API
* Overview of 10 Python Clustering Methods
* Clustering Taxonomy
* K-means Clustering

# K-means Python demo & clustering considerations 
* Demo
* K-means Pro and Con
* Algorithm Comparison

# Hierarchical clustering and k-means / hierarchical clustering Python file
* Linkage Methods
* Threshold Determination

# Principal components analysis & anomaly detection and anomaly detection Python file 
* Computational Complexity
* "Scree" plots
* Anomaly Detection Application

---
# Week 4
https://aueems.cce.af.mil/courses/9884/pages/week-4-overview
## Readings
* RAND report - DoD AI Posture Download RAND report - DoD AI Posture(skim) 
* ISLP & HOML readings assigned on last page of syllabus

## Videos
### Best stepwise selection
* Brute force approach
* Comparing models - Cp, AIC, BIC, Adjusted R2
* Python file: 4 Week 4A feature_selection_expanded with BIC and RFE v3.ipynbDownload 4 Week 4A feature_selection_expanded with BIC and RFE v3.ipynb
    * Please note: in the video this file has the name "feature selection expanded.ipynb"
* Python dataset: UScrime 2.csvDownload UScrime 2.csv
### Forward & backward stepwise selection (amended) 
* Method
* Computational efficiencies
* Python file and dataset: same as above (Week 4A...)
### L1 (Lasso) & L2 (Ridge) Regularization 
* Penalty functions
* Impact on coefficients
* Bias / Variance tradeoff
###  L1 & L2 Regularization Python demo
* Python file: 4 Ridge_and_Lasso_from_ISLR_Chapter_6 v2.ipynbDownload 4 Ridge_and_Lasso_from_ISLR_Chapter_6 v2.ipynb
* Python dataset: Hitters datasets.zipDownload Hitters datasets.zip
### Dimension reduction with Principal Components Regression (PCR)
* Theory
* Python file and dataset: same as above (Ridge_and_Lasso...)
### Dimension reduction with Partial Least Squares Regression (PLS)
* Comparison with PCR
* Literature example
* Python file and dataset: same as above (Ridge_and_Lasso...)
### Natural language processing - NLP - Feature selection on the English language
* Python file: 4G_NLP_demo_v4.ipynbDownload 4G_NLP_demo_v4.ipynb
* Python dataset: NLP Training data.csv

## Lecture slides

Combined  3.week 4 slidesDownload Combined week 4 slides
Natural language processing slidesDownload Natural language processing slides
 

## Bonus material - NLP

A few students in prior sections used NLP for their analysis, either on its own, or in addition to numeric/categorical variables, and I've attached a few of their papers.  NLP is fairly advanced and if you are feeling very comfortable with python and general machine learning, it could be a good way to enhance your learning experience.  It definitely isn't required in this course.  

NLP TMT DASC 522 Final Paper.docx  (just NLP)

NLP Audit cycle time DASC 522 Final Paper.docx (NLP, numeric & categorical variables)
 

## Bonus material - plot format

It can take a lot of time to make your figures look good (and be legible) in a two-column  journal template for the final project. You'll need to adjust various font sizes, the background color and other attributes.
In an effort to save you time, I've attached a python script that looks good in a two-column format. It consolidates a few stack overflow posts on the topic, and allows you to easily fine-tune your figures.
Standard plot format.ipynb


# Week 5
## Reading

* ISLP & HOML readings assigned on last page of syllabus
* DASC 522 Guide to Hyperparameter (HP) Selection
    * This guide summarizes the main neural network HPs that you'll learn about in the next 5 weeks, and offers recommendations for what values to initially select based on your problem type and other factors.  
    * However, since every dataset is unique, numerous departures from these starting points may be required to find the best performance. The guide is organized into 3 areas:
        * Structure: the HPs you set when you create your NN with model = keras.Sequential( ...layers neurons blah blah blah )
        * Fit:            the HPs you set when you use model.fit( ...code here )
        * Compile:   the HPs you set when you use model.compile( ...code here )
 

## Lecture videos

### NN foundations 
* Introduction to Keras
* Python demo - Pima Indians Classification (*not "regression" per the python file title)
* NN visualization
* Overfitting


### Perceptron overview
* NN structure & types of learning
* Activation functions
* Difference between perceptron vs. NN
* Backpropagation

### Perceptrons & perceptron learning
* Perceptron components
* Application to logic problems
* Perceptron learning vs. gradient descent

### Multilayer perceptrons (a.k.a. NN)
* Structure vs performance
* Hyperparameter introduction
* Notable example NNs: image recognition, Siri, AlphaGo, self-driving car
 

## Lecture slides & python files

* NN foundations & Perceptron overview
    * 5A Pima Indian classification.ipynb  and dataset
* Perceptrons & perceptron learning
* Multilayer perceptrons and excel calculation file
 

## Bonus material - plotting histograms of numeric variables

Here is a code snippet to create the histograms that are shown in the Homework #4 project template. 

The histograms use the Seaborn library (sns), which is an extension of python's Matplotlib plotting library. 
```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

sns.set_theme(style='darkgrid')  # set plotting style
df2 = df[['THC', 'Age', 'Education', 'Nscore', 'Escore', 'Oscore', 'Ascore','Cscore', 'Impulsive', 'SS']]  # 10 variables in the dataframe
axes = df2.plot.hist(subplots=True, 
                     layout=(5, 2),  # 5 x 2 grid of plots
                     figsize=(12,12), 
                     legend=True,
                     cmap='viridis',  # colormap to look nice
                     fontsize=12)
```

---
# Week 6 Overview
## Readings
* ISLP & HOML readings assigned on last page of syllabus
 
## Lecture videos (Python demonstrations in all)
### Backpropagation
* Fundamentals
* Batch Size
* Widrow's Rule of Thumb for total neuron count: Widrow ROT Number of points required.xlsx 
* "Early stopping" to improve performance

### Optimization part 1 
* Loss functions
* Optimizer algorithms
* Learning parameters

### Optimization part 2 
* Batch normalization
* Weight & bias initialization
* The impact of feature normalization on NN performance

### Resampling part 1 
* Train/validation/test split
* Leave-one-out cross-validation (LOOCV)

### Resampling part 2 
* k-fold cross validation
* k-fold vs LOOCV
* Bias & variance tradeoff

## Python files
* 6A Early Stopping.ipynb
* 6B Optimization v2.ipynb 
* 6C_Resampling_&_Cross_Validation_v3.ipynb 
* Lectures refer again to the 5A Pima Indian classification.ipynb & dataset - they are located in Week 5

## Lecture slides
* Backpropagation
* Combined Optimization
* Combined Resampling
 

## Bonus material

A "must" when applying machine learning techniques to real problems (i.e. your final project) is to validate your models.  Without this key step, your model may give the impression of accuracy, but then fail to perform when it is first used.

For classical techniques (regression, logistic regression, decision trees etc) a two-way train/holdout split is good.

For neural networks a 3 way split is needed, since the training routine uses both the train & test sets, and (generally) a large number of sweeps and models are created.  This can result in overfitting of the test set, requiring a third (holdout) set to validate the model.  Ideally, after creating a large number of NN models, you'll select only a handful to check against the holdout set.

There are lots of ways to do this, but here are some. In my projects I tend to use the #1 two-way split for classical, and then use those same datasets in #4 for NN:

### 1) Two-way split - split after create X/y - 70/30 split. THC is the label
```python
from sklearn.model_selection import train_test_split
X = df2.loc[:, df2.columns != 'THC']
y = df2.loc[:, df2.columns == 'THC']

X = sm.add_constant(X)   # only needed for sm (not smf)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=0) #random state is a seed value used to reproduce your split later

print(X_train.shape, X_val.shape)
print(y_train.shape, y_val.shape)
```

### 2) Two-way split - split before create X/y - 70/30 split. Length is the label
```python
# split into 2 dataframes
df_train=df.sample(frac=0.8,random_state=42) 
df_val=df.drop(train.index)

# model on train set
model=smf.ols('Length~Width', df_train).fit()
print(model.summary2())

# split train/test into X_train, y_train & X_test, y_test
X_train = df_train['Width']
y_train = df_train['Length']
X_val = df_val['Width']
y_val = df_val['Length']

 
# compare MSE for train and test sets
y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)
RMSE_train = mean_squared_error(y_train, y_pred_train)
RMSE_val = mean_squared_error(y_val, y_pred_val)
 ```

### 3) Three-way split - separate variables - this gives a 70/15/15 split. y is the label
```python
from sklearn.model_selection import train_test_split
# 1st split - remove 15% for validation (holdout)
X_splitAgain, X_val, y_splitAgain, y_val = train_test_split(X, y, test_size=0.15, random_state=42) 

# 2nd split
X_train, X_test, y_train, y_test = train_test_split(X_splitAgain, y_splitAgain, test_size=0.1763, random_state=42) 

print('Train set shape\n',X_train.shape, y_train.shape)
print('Test set shape\n',X_test.shape, y_test.shape)
print('Val set shape\n',X_val.shape, y_val.shape)
print(' ')
 ```
### 4) Three way split using built-in "validation_split" in NN fitting function  
a) split data into train & validation/holdout sets using a two-way method from #1 above  
b) use the validation_split method of .fit to automatically split the training dataset into train & test datasets 
```python
history = dnn_model.fit(train_features, 
                        train_labels,
                        validation_split=0.2,
                        epochs=100)
```


---
# Week 7 Overview
## Readings
* HOML reading assigned on last page of syllabus
* Journal article: Types of minority class examples and their influence on learning classifiers from imbalanced dataDownload Types of minority class examples and their influence on learning classifiers from imbalanced data
 

## Lecture videos (all are Python demonstrations)
### Neural Network (NN) Regression part 1 Download Neural Network (NN) Regression part 1 
* Demonstration that the weight & bias of a single-perception NN match the y=mx+b prediction of classical linear regression
* Single-variable regression using conventional & deep NNs
## NN Regression part 2 Download NN Regression part 2 (updated)
* Multiple-variable regression using conventional & deep NNs
* Demonstration of how normalization is essential for NN training
* Comparison to classical linear regression
### Neural Network ClassificationDownload Neural Network Classification
* Accuracy, precision & recall
* Threshold variation, ROC curve & AUC
* Introduction to hyperparameter tuning using classification threshold
### Hyperparameter tuning Download Hyperparameter tuningof model structure, compilation & fit parameters via 3 methods
* Hand-code
* Scikit-learn GridSearchCV
* “Weights & Biases” framework interface
* Hyperparameter tuning concept map.jpg

 
## Lecture Python files
* There are no powerpoint files this week
* 7A NN Regression tensorflow example v2.ipynb Download 7A NN Regression tensorflow example v2.ipynb   7A NN Regression tensorflow example.ipynb
* 7B GMLC Binary Classification v2.ipynbDownload 7B GMLC Binary Classification v2.ipynb
* 7C v2 NN Hyperparameter Search - Classification.ipynbDownload 7C v2 NN Hyperparameter Search - Classification.ipynb
* 5A pima-indians-diabetes.data.csvDownload 5A pima-indians-diabetes.data.csv
 
## Bonus material
There are two main ways of setting up your model & layers:

1. (recommended) Using the TensorFlow.Keras methods featured in the class demos. Any categorical inputs need to be one-hot encoded prior to modeling, as the model assumes all input have a numeric datatype (float/int/boolean).  The TensorFlow certification that I earned uses these methods, and I've used these methods for all of my machine learning projects.
2. (not recommended) Using the TensorFlow  feature_column/feature_layer methods to specify datatypes prior to modeling.  I have heard this is often used when setting up a web-facing end-to-end machine learning pipeline.  The process is described in this article Download this article. 
 
Here are code snippets to show the differences between the methods:
1)  ------------------ used in class demos
```python
normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))
number_of_inputs= train_features.shape[1]
model = keras.Sequential([normalizer,
                          layers.Dense(16, activation='relu', input_dim=number_of_inputs),
                          layers.Dense(16, activation='relu'),
                          layers.Dense(1, activation='linear') ]) # output layer for regression

model.compile(...
history = model.fit(...
```

2) ---------------- Tensorflow feature_column/feature_layer notation
* Create a utility method to convert your dataframe into a tf.data dataset
* Create an input pipeline
* For each feature column
    * Create a feature_column based on its datatype (and add normalizer for each) based on https://www.tensorflow.org/tutorials/structured_data/feature_columns (Links to an external site.)
    * Append that feature to a TensorFlow feature_layer
```python
model = keras.Sequential([feature_layer,
                          layers.Dense(16, activation='relu', input_dim=number_of_inputs),
                          layers.Dense(16, activation='relu'),
                          layers.Dense(1, activation='linear') ]) # output layer for regression

model.compile(...
history = model.fit(...
```