import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score,hamming_loss
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_validate, train_test_split, ShuffleSplit, GridSearchCV, learning_curve
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import statistics
import warnings
warnings.filterwarnings("ignore")
#Load Data
traindf=pd.read_csv('C:\\Users\\Mohamad\\Desktop\\v\\vad_hlex_lex_doc2vec_freq_bi_tweets_train.csv')
devdf=pd.read_csv('C:\\Users\\Mohamad\\Desktop\\v\\vad_hlex_lex_doc2vec_freq_bi_tweets_dev.csv')
testdf=pd.read_csv('C:\\Users\\Mohamad\\Desktop\\v\\vad_hlex_lex_doc2vec_freq_bi_tweets_test.csv')
traindf.columns.values
#Define Model features
feature_vector = ["freq_anger","bi_anger","doc2vec_anger","lex_Anger","hlex_anger", "freq_anticipation","bi_anticipation","doc2vec_anticipation", "lex_Anticipation", "hlex_anticipation", "freq_disgust","bi_disgust","doc2vec_disgust","lex_Disgust", "hlex_disgust", "freq_fear","bi_fear","doc2vec_fear", "lex_Fear", "hlex_fear", "freq_joy","bi_joy","doc2vec_joy", "lex_Joy","hlex_joy", "freq_love", "bi_love", "doc2vec_love", "freq_optimism", "bi_optimism", "doc2vec_optimism", "freq_pessimism", "bi_pessimism", "doc2vec_pessimism", "freq_sadness","bi_sadness","doc2vec_sadness", "lex_Sadness", "hlex_sadness", "freq_surprise","bi_surprise","doc2vec_surprise","lex_Surprise", "hlex_surprise", "freq_trust","bi_trust","doc2vec_trust", "lex_Trust", "hlex_trust", "lex_Positive","lex_Negative","V","D","A"]
train_x = traindf[feature_vector]
dev_x = devdf[feature_vector]
test_x = testdf[feature_vector]
#feature vector length
len(train_x.columns)
#define emotion class
em=["anger","anticipation","disgust","fear","joy","love","optimism","pessimism","sadness","surprise","trust"]
#reformate emotion classes
def labelling_classes(df):
    arr=[]
    for i,val in enumerate(df.iterrows()):
        lbl=[]
        for e in em:
            if df.iloc[i][e]==1:
                lbl.append(e)
        arr.append(lbl)
    return arr
#define classes 
train_y=traindf[em]
train_y["classes"]=np.array(labelling_classes(traindf))
dev_y=devdf[em]
dev_y["classes"]=np.array(labelling_classes(devdf))
#split data for training and testing
#shuffle to apply random shuffle on data splitting
cv = ShuffleSplit( n_splits=5, test_size=0.33)
mlb = MultiLabelBinarizer(classes=("anger","anticipation","disgust","fear","joy","love","optimism","pessimism","sadness","surprise","trust"))
y_enc = mlb.fit_transform(train_y["classes"])

mlb = MultiLabelBinarizer(classes=("anger","anticipation","disgust","fear","joy","love","optimism","pessimism","sadness","surprise","trust"))
ydev_enc = mlb.fit_transform(dev_y["classes"])
 #hamming score 
def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    http://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/float(len(set_true.union(set_pred)) )
        acc_list.append(tmp_a)
    return np.mean(acc_list)
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
from matplotlib import pyplot
def return_counts(df,x):
    total=0
    for e in em:
        total = total + len(df[df[e]==x])
    return total
    
def plot_test_res(df,predict):
    i=0
    for e in em:
        df[e]=predict[:,i]
        i = i +1
    
    ax,fig=pyplot.subplots(len(em)+1, 1,figsize=(15,25))
    i=0
    for e in em:
        i=i+1
        pyplot.subplot(len(em)+1,1,i)
        df[e].value_counts().plot.bar(title=e)
    
    ones = return_counts(df,1)
    print("ones",ones)
    zeros = return_counts(df,0)
    print("zeros",zeros)
    pyplot.subplot(len(em)+1,1,len(em)+1)
    pyplot.bar(["1","0"],[ones,zeros])
    pyplot.show()   
    
#Support vector machine
cv = ShuffleSplit( n_splits=5, test_size=0.25)
svm = BinaryRelevance(classifier = SVC(probability=True))
param_values = {'classifier__gamma': [1, 0.01]}  
GS_svm = GridSearchCV(svm, param_grid=param_values, cv=cv)
GS_svm.fit(train_x,y_enc)
print("best estimator parameters",GS_svm.best_estimator_)
print("training data evaluation")
y_pred=GS_svm.best_estimator_.predict(train_x)
score=hamming_score(y_enc,y_pred.toarray())
loss=hamming_loss(y_enc,y_pred.toarray())
print("hamming score",score)
print("hamming loss",loss)
print(classification_report(y_enc,y_pred))
print("development data evaluation")
y_pred=GS_svm.best_estimator_.predict(dev_x)
score=hamming_score(ydev_enc,y_pred.toarray())
loss=hamming_loss(ydev_enc,y_pred.toarray())
print("hamming score",score)
print("hamming loss",loss)
print(classification_report(ydev_enc,y_pred))
print("plot learning curve")
plot_learning_curve(GS_svm.best_estimator_, "SVM", train_x,y_enc, cv=cv)

#test data
test_pred=GS_svm.best_estimator_.predict(test_x).toarray()
plot_test_res(testdf, test_pred)