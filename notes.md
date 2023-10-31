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



