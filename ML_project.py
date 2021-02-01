# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 10:32:54 2021

@author: monob
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import  preprocessing
from scipy.stats.mstats import winsorize
import seaborn as sns


#########################################################################################

############################        LOAD DATA             ###############################

#########################################################################################

df = pd.read_csv('bank-train.csv')
df_test = pd.read_csv('bank-test.csv')
df_validation = pd.read_csv('bank-validation.csv')

columns_name = ['Age','Type of job','Marital status','education','credit in default','balance bank',
                                   'has housing loan?','has personal loan?','contact communication','contact day of the month','contact month of year ',
                                  'contact duration','number of contacts performed during','number of days that passed','number of contacts performed',
                                   'outcome of the previous marketing',
                                  'client subscribed']

print(df.head(3))

print(df.shape)

# RENAME COLUMNS

df.columns = columns_name
df_test.columns = columns_name
df_validation.columns = columns_name

print(df.dtypes)

df[['Type of job','Marital status','education','credit in default',
                                   'has housing loan?','has personal loan?','contact communication','contact month of year ',
                                   'outcome of the previous marketing',
                                  'client subscribed']] = df[['Type of job','Marital status','education','credit in default',
                                   'has housing loan?','has personal loan?','contact communication','contact month of year ',
                                   'outcome of the previous marketing',
                                  'client subscribed']].astype(str)

df_test[['Type of job','Marital status','education','credit in default',
                                   'has housing loan?','has personal loan?','contact communication','contact month of year ',
                                   'outcome of the previous marketing',
                                  'client subscribed']] = df_test[['Type of job','Marital status','education','credit in default',
                                   'has housing loan?','has personal loan?','contact communication','contact month of year ',
                                   'outcome of the previous marketing',
                                  'client subscribed']].astype(str)
df_validation[['Type of job','Marital status','education','credit in default',
                                   'has housing loan?','has personal loan?','contact communication','contact month of year ',
                                   'outcome of the previous marketing',
                                  'client subscribed']] = df_validation[['Type of job','Marital status','education','credit in default',
                                   'has housing loan?','has personal loan?','contact communication','contact month of year ',
                                   'outcome of the previous marketing',
                                  'client subscribed']].astype(str)
                                                                         
#########################################################################################

############################       DATA ANALYSIS        #################################

#########################################################################################                                                                         
                                                                         
                                                                         
                                                                         
print(df.info())                                                                         
print(df.describe())
print(df.isnull().sum())

duplicate = df.duplicated()
print(duplicate.sum())

#unique values for each column
for i in df.columns:
    print(i)
    print(df[i].unique())
    print('---'*20)

# Function to detect outliers in every feature
def detect_outliers(dataframe):
    cols = list(dataframe)
    outliers = pd.DataFrame(columns = ['Feature', 'Number of Outliers'])
    for column in cols:
        if column in dataframe.select_dtypes(include=np.number).columns:
            q1 = dataframe[column].quantile(0.25)
            q3 = dataframe[column].quantile(0.75)
            iqr = q3 - q1
            fence_low = q1 - (1.5*iqr)
            fence_high = q3 + (1.5*iqr)
            outliers = outliers.append({'Feature':column, 'Number of Outliers':dataframe.loc[(dataframe[column] < fence_low) | (dataframe[column] > fence_high)].shape[0]},ignore_index=True)
    return outliers

print(detect_outliers(df))


### numerical
numerical_cols = list(df.select_dtypes(exclude=['object']))
print(numerical_cols)

### categorical
category_cols = list(df.select_dtypes(include=['object']))
print(category_cols)


print(df['client subscribed'].value_counts())

# see the balance of  the classes
def class_imbalance(target):
    class_values = (target.value_counts()/target.value_counts().sum())*100
    return class_values

print(class_imbalance(df['client subscribed']))

df['client subscribed'].hist()

# Function to plot categorical variables
def plot_categorical(dataframe):
    categorical_columns = dataframe.select_dtypes(include=['object']).columns
    
    for i in range(0,len(categorical_columns),2):
            if len(categorical_columns) > i+1:
                
                plt.figure(figsize=(10,4))
                plt.subplot(121)
                dataframe[categorical_columns[i]].value_counts(normalize=True).plot(kind='bar')
                plt.title(categorical_columns[i])
                plt.subplot(122)     
                dataframe[categorical_columns[i+1]].value_counts(normalize=True).plot(kind='bar')
                plt.title(categorical_columns[i+1])
                plt.tight_layout()
                plt.show()

            else:
                dataframe[categorical_columns[i]].value_counts(normalize=True).plot(kind='bar')
                plt.title(categorical_columns[i])
        
        
plot = plot_categorical(df)
plt.show()

# Function to plot continuous variables
def plot_continuous(dataframe):
    numeric_columns = dataframe.select_dtypes(include=['number']).columns.tolist()
    dataframe = dataframe[numeric_columns]
    
    for i in range(0,len(numeric_columns),2):
        if len(numeric_columns) > i+1:
            plt.figure(figsize=(10,4))
            plt.subplot(121)
            sns.distplot(dataframe[numeric_columns[i]], kde=False)
            plt.subplot(122)            
            sns.distplot(dataframe[numeric_columns[i+1]], kde=False)
            plt.tight_layout()
            plt.show()

        else:
            sns.distplot(dataframe[numeric_columns[i]], kde=False)

plot = plot_continuous(df)
plt.show()

def target_categorical(dataframe,target):
    categorical_columns = dataframe.select_dtypes(exclude=np.number).columns
    for i in range(0,len(categorical_columns),2):
        if len(categorical_columns) > i+1:
            plt.figure(figsize=(15,5))
            plt.subplot(121)
            sns.countplot(x=dataframe[categorical_columns[i]],hue=target,data=dataframe)
            plt.xticks(rotation=90)
            plt.subplot(122)            
            sns.countplot(dataframe[categorical_columns[i+1]],hue=target,data=dataframe)
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.show()


plot = target_categorical(df,df['client subscribed'])
plt.show()

df['contact duration'] = df['contact duration'].apply(lambda n:n/60).round(2)
df_test['contact duration'] = df_test['contact duration'].apply(lambda n:n/60).round(2)
df_validation['contact duration'] = df_validation['contact duration'].apply(lambda n:n/60).round(2)

duration_campaign = sns.scatterplot(x='contact duration', y='number of contacts performed during',data = df,
                     hue = 'client subscribed')

plt.axis([0,65,0,65])
plt.ylabel('Number  Calls')
plt.xlabel('Duration Calls ')
plt.title(' Number and Duration of Calls')
# Annotation
plt.show()


#########################################################################################

############################        DATA PREPROCESSING       ############################

#########################################################################################


# Function to treat outliers 
def treat_outliers(dataframe):
    cols = list(dataframe)
    for col in cols:
        if col in dataframe.select_dtypes(include=np.number).columns:
            dataframe[col] = winsorize(dataframe[col], limits=[0.05, 0.1],inclusive=(True, True))
    
    return dataframe    


df = treat_outliers(df)
df_test = treat_outliers(df_test)
df_validation = treat_outliers(df_validation)



def convertcategory(df):
    '''
    Convert the categorical dataframe into numerical dataframe
    
    
    '''
    imp_df = df

    for f in imp_df.columns:
        if imp_df[f].dtype== 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(imp_df[f].values))
            imp_df[f] = lbl.transform(list(imp_df[f].values))
            
    return imp_df

#preprocessing
df = convertcategory(df)
df_test = convertcategory(df_test)
df_validation = convertcategory(df_validation)

#scaling features
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

features = df[numerical_cols]

ct = ColumnTransformer([
        ('somename', StandardScaler(), numerical_cols)
    ], remainder='passthrough')


features = ct.fit_transform(features)
df[numerical_cols] = features


features2 = df_test[numerical_cols]


features2 = ct.fit_transform(features2)
df_test[numerical_cols] = features2

features3 = df_validation[numerical_cols]


features3 = ct.fit_transform(features3)
df_validation[numerical_cols] = features3


def CorrData(input_df):

 
    sns.heatmap(df[numerical_cols].corr(),annot=True,cmap = 'Blues',linewidths=0.2) #data.corr()-->correlation matrix
    fig=plt.gcf()
    fig.set_size_inches(18,8)
    plt.show()

# CORRELATION AMONG DATA
    
plot = CorrData(df[numerical_cols]) 
plt.show()

#########################################################################################

############################        FEATURE SELECTION       ###############################

#########################################################################################

x_train = df.drop(['client subscribed'], axis = 1)
y_train = df['client subscribed']

x_test = df_test.drop(['client subscribed'], axis = 1)
y_test = df_test['client subscribed']

x_validation = df_validation.drop(['client subscribed'], axis = 1)
y_validation = df_validation['client subscribed']


print("predictor of input dataset shape: {shape}".format(shape=x_train.shape))
print("predictor of target dataset shape: {shape}".format(shape=y_train.shape))

print("predictor of input dataset shape: {shape}".format(shape=x_validation.shape))
print("predictor of target dataset shape: {shape}".format(shape=y_validation.shape))

print("predictor of input dataset shape: {shape}".format(shape=x_test.shape))
print("predictor of target dataset shape: {shape}".format(shape=y_test.shape))

import statsmodels.api as sm
X_1 = sm.add_constant(x_train)
#Fitting sm.OLS model
model = sm.OLS(y_train,X_1).fit()
model.pvalues

#Backward Elimination
cols = list(x_train.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = x_train[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y_train,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print("Feature selection ", selected_features_BE)

x_train = x_train[selected_features_BE]
x_validation = x_validation[selected_features_BE]
x_test = x_test[selected_features_BE]


############################        CLASS BALANCE      ##################################

#########################################################################################

# balancing of the target variable

from imblearn.over_sampling import RandomOverSampler
oversample = RandomOverSampler(sampling_strategy='minority')

# fit and apply the transform
x_train, y_train = oversample.fit_resample(x_train, y_train)

y_train.hist()


############################        MODEL SELECTION      ################################

#########################################################################################


############################        NAIVE BAYES      ####################################

#########################################################################################

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import sklearn.metrics as metrics
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import plot_confusion_matrix
bnb = BernoulliNB(binarize=0.0)
bnb.fit(x_train, y_train)
acc_train = bnb.score(x_train,y_train)*100


y_pred = bnb.predict(x_validation)

cm = confusion_matrix(y_validation, y_pred)

acc = accuracy_score(y_validation, y_pred)*100
pre = precision_score(y_validation, y_pred)
recall = recall_score(y_validation, y_pred)
f1 = f1_score(y_validation, y_pred)


print("Accuracy Train Naive Bayes :",acc_train)
print("Accuracy Test  Naive Bayes :",acc)
print("Confusion Matrix : \n",cm)
print("Precision :",pre)
print("Recall :",recall)
print("F1-score :",f1)
plot_confusion_matrix(bnb, x_validation, y_validation,cmap=plt.cm.Blues)



############################       LOGISTIC REGRESSION       ############################

#########################################################################################


# standard con max_iter = 100 e C = 1.0
from sklearn.linear_model import LogisticRegression


lr = LogisticRegression(max_iter = 10000)
lr.fit(x_train, y_train)

acc_train = lr.score(x_train,y_train)*100


y_pred = lr.predict(x_validation)

cm = confusion_matrix(y_validation, y_pred)

acc = accuracy_score(y_validation, y_pred)*100
pre = precision_score(y_validation, y_pred)
recall = recall_score(y_validation, y_pred)
f1 = f1_score(y_validation, y_pred)


print("Accuracy Train logistic standard:",acc_train)
print("Accuracy Test logistic standard:",acc)
print("Confusion Matrix : \n",cm)
print("Precision :",pre)
print("Recall :",recall)
print("F1-score :",f1)

plot_confusion_matrix(lr, x_validation, y_validation,cmap=plt.cm.Reds)


######################    LOGISTIC REGRESSION  PARAMETER     ############################

#########################################################################################

# find the optimal parameter for logistic regression
C = [ 0.00000001,0.00001,0.01, 0.1, 1, 10, 100, 1000]
best_accuracy = 0
best_train = 0
best_C = 0
best_recall = 0
for i in C:
    lr_1 = LogisticRegression(C = i, max_iter = 100000)
    lr_1.fit(x_train, y_train)
    y_pred_train = lr_1.predict(x_train)
    y_pred = lr_1.predict(x_validation)
    acc = accuracy_score(y_validation, y_pred)*100
    acc_train = lr.score(x_train,y_train)*100
    pre = precision_score(y_validation, y_pred)
    recall = recall_score(y_validation, y_pred)
    f1 = f1_score(y_validation, y_pred)
    if recall > best_recall:
            best_accuracy = acc
            best_train = acc_train
            best_C = i
            best_lr = lr_1
            best_recall = recall
print('The accuracy train of the logistic regression model is: ', best_train, ' C : ', best_C)
print('The accuracy test of the logistic regression model is: ', best_accuracy, ' C : ', best_C)
print('The recall of the model is: ', best_recall, ' C : ', best_C)
plot_confusion_matrix(best_lr, x_validation, y_validation,cmap=plt.cm.Greens)
print('   ')

######################            SVM LINEAR   KERNEL         ###########################

#########################################################################################

from sklearn.svm import SVC

# find the optimal parameter for linear kernel
C = [0.000001,0.00001, 0.01, 0.1, 1]
best_accuracy = 0
best_train = 0
best_C = 0
best_recall = 0
for i in C:
    svc = SVC(C = i, kernel = 'linear')
    svc.fit(x_train, y_train)
    y_pred_train = svc.predict(x_train)
    y_pred = svc.predict(x_validation)
    acc = accuracy_score(y_validation, y_pred)*100
    acc_train = svc.score(x_train,y_train)*100
    pre = precision_score(y_validation, y_pred)
    recall = recall_score(y_validation, y_pred)
    f1 = f1_score(y_validation, y_pred)
    if recall > best_recall:
            best_accuracy = acc
            best_train = acc_train
            best_recall = recall
            best_C = i
            best_svc = svc 
print('The accuracy train of the SVC linear kernel model is: ', best_train, ' C : ', best_C)
print('The accuracy test of the SVC linear kernel model is: ', best_accuracy, ' C : ', best_C)
print('The recall of the model is: ', best_recall, ' C : ', best_C)
plot_confusion_matrix(best_svc, x_validation, y_validation,cmap=plt.cm.Greys)
print('   ')

######################         SVM   RADIO KERNEL         ###############################

#########################################################################################


from sklearn.svm import SVC

# find the optimal parameter for radio kernel
C = [0.000001,0.00001, 0.01, 0.1, 1, 10]
best_accuracy = 0
best_train = 0
best_C = 0
best_gamma = 0
best_recall = 0
gamma = [1,0.1,0.001,0.0001, 0.00000000000001]
for i in C:
    for g in gamma:
        svc = SVC(C = i, kernel = 'rbf', gamma = g)
        svc.fit(x_train, y_train)
        y_pred_train = svc.predict(x_train)
        y_pred = svc.predict(x_validation)
        acc = accuracy_score(y_validation, y_pred)*100
        acc_train = svc.score(x_train,y_train)*100
        pre = precision_score(y_validation, y_pred)
        recall = recall_score(y_validation, y_pred)
        f1 = f1_score(y_validation, y_pred)
        if  acc > best_accuracy:
            best_accuracy = acc
            best_C = i
            best_train = acc_train
            best_recall = recall
            best_gamma = g
            best_svc = svc
            
            
print('The accuracy train of the SVM radio kernel model is: ', best_train, ' C : ', best_C, 'gamma : ', best_gamma)    
print('The accuracy test of the SVM radio kernel model is: ', best_accuracy, ' C : ', best_C, 'gamma : ', best_gamma)
print('The recall of the model is: ', best_recall, ' C : ', best_C)
plot_confusion_matrix(best_svc, x_validation, y_validation,cmap=plt.cm.Oranges)
print('   ')                                                        

######################         BEST MODEL         #######################################

#########################################################################################                                                                         
                                                                         

#BEST MODEL
svc = SVC(C = 10, kernel = 'rbf', gamma = 0.1)
svc.fit(x_train, y_train)
y_pred_test = svc.predict(x_test)
acc = accuracy_score(y_test, y_pred)*100
pre = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy Test set SVM radio kernel:",acc)
print("Confusion Matrix : \n",cm)                                                                         
                                                                         
                                                                         
                                                                         
                                                                         
                                                                         
                                                                         
                                                                         
                                                                         
                                                                         
                                                                         
                                                                         