import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy import stats
from tabulate import tabulate
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, confusion_matrix, precision_recall_curve, roc_curve,roc_auc_score,precision_recall_curve, auc, f1_score
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

import warnings
warnings.filterwarnings('ignore')

dataset = pd.read_csv("employee data.csv")
pd.options.display.max_columns = None
dataset.head()

dataset.isnull().any()

dataset.describe().T

f, ax = plt.subplots(2,6,figsize = (25,25))
sns.boxplot(data = dataset['PercentSalaryHike'],ax = ax[0][0]).set_title('PercentSalaryHike')
sns.boxplot(data = dataset['TotalWorkingYears'],ax = ax[0][1]).set_title('TotalWorkingYears')
sns.boxplot(data = dataset['TrainingTimesLastYear'], ax = ax[0][2]).set_title('TrainingTimesLastYear')
sns.boxplot(data = dataset['YearsAtCompany'],ax = ax[0][3]).set_title('YearsAtCompany')
sns.boxplot(data = dataset['YearsInCurrentRole'],ax = ax[0][4]).set_title('YearsInCurrentRole')
sns.boxplot(data = dataset['YearsSinceLastPromotion'],ax = ax[0][5]).set_title('YearsSinceLastPromotion')
sns.boxplot(data = dataset['YearsWithCurrManager'],ax = ax[1][0]).set_title('YearsWithCurrManager')
sns.boxplot(data = dataset['Age'],ax = ax[1][1]).set_title('Age')
sns.boxplot(data = dataset['DailyRate'],ax = ax[1][2]).set_title('DailyRate')
sns.boxplot(data = dataset['DistanceFromHome'],ax = ax[1][3]).set_title('DistanceFromHome')
sns.boxplot(data = dataset['MonthlyIncome'],ax = ax[1][4]).set_title('MonthlyIncome')
sns.boxplot(data = dataset['NumCompaniesWorked'],ax = ax[1][5]).set_title('NumCompaniesWorked')

# Dropping Redundant Columns
dataset.drop(['EmployeeNumber','EmployeeCount', 'Over18', 'StandardHours'] ,axis = 1,inplace=True)

# Plotting the correlation matrix
f, ax = plt.subplots(figsize=(20,20))
corr = dataset.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax, annot=True)


# Converting the Attrition, Overtime and Gender columns into binary integers
dataset['Attrition'] = dataset['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0) 
dataset['OverTime'] = dataset['OverTime'].apply(lambda x: 1 if x == 'Yes' else 0) 
dataset['Gender'] = dataset['Gender'].apply(lambda x: 1 if x == 'Male' else 0)

# Combining EnvironmentSatisfaction, JobInvolvement, JobSatisfaction, RelationshipSatisfaction to form a new Feature HolisticSatisfaction
dataset['HolisticSatisfaction'] = dataset['EnvironmentSatisfaction'] + dataset['JobInvolvement'] + dataset['JobSatisfaction'] + dataset['RelationshipSatisfaction']

# Dropping The Original Satisfaction Features
dataset.drop(['JobInvolvement','JobSatisfaction','RelationshipSatisfaction','EnvironmentSatisfaction'],axis = 1)
print(dataset.shape)
dataset.head()

# Plotting the Correlation Matrix Again with updated set of features
f, ax = plt.subplots(figsize=(20,20))
corr = dataset.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax, annot=True)


# Plotting the distribution of the attrition
sns.countplot(x ='Attrition',data = dataset)

# Plotting Age and Attrition
sns.catplot(x='Age', y='Attrition', aspect=5, kind='bar', data=dataset)

# Plotting Deparment and Attrition
sns.catplot(x = 'Department' , y = 'Attrition', aspect = 2, kind = 'bar', data = dataset)
# Plotting Job Role and Attrition
sns.catplot(x = 'JobRole', y = 'Attrition', aspect = 3 , kind = 'bar', data  = dataset)

# Plotting Gender and Attrition
sns.catplot(x = 'Gender', y = 'Attrition', aspect = 1, kind = 'bar', data = dataset)

# Plotting MaritalStatus and Attrition
sns.catplot(x = 'MaritalStatus', y = 'Attrition', aspect = 2 , kind = 'bar', data = dataset)

# Plotting OverTime and Attrition
ax = sns.barplot(x="OverTime", y="MonthlyIncome", hue="Attrition", data=dataset, estimator=lambda x: len(x) / len(dataset) * 100)
ax.set(ylabel="Percent")

# Plotting HolisticSatisfaction and Attrition
sns.catplot(x = 'HolisticSatisfaction',y = 'Attrition', aspect = 3, kind = 'bar', data = dataset)

# Plotting Monthly Income and Attrition
fig = plt.figure(figsize=(15,4))
ax=sns.kdeplot(dataset.loc[(dataset['Attrition'] == 0),'MonthlyIncome'] , color='b',shade=True, label='No Attrition')
ax=sns.kdeplot(dataset.loc[(dataset['Attrition'] == 1),'MonthlyIncome'] , color='r',shade=True, label='Attrition')
plt.title('Employee Monthly Income Distribution - Attrition V.S. No Attrition')

# Dummy Variables are generated for categorical features
dataset = pd.get_dummies(data = dataset, columns=['BusinessTravel','Department','JobRole','MaritalStatus','EducationField'])
dataset.head()

# Z score is also called standard score.
# This score helps to understand if a data value is greater or smaller than mean and how far away it is from the mean.
# Z score tells how many standard deviations away a data point is from the mean.

cols = list(dataset.columns)
remove = []
dt = np.array(dataset)
thresh = 2.5

# Calculating ZScore and removing outliers from YearsAtCompany, TotalWorkingHours, and MonthlyIncome
for i in [13, 23, 26]:
    data = dt.T[i]
    m = np.mean(data)
    s = np.std(data)
    for i in range(len(data)):
        z = abs((data[i] - m) / s)
        if(z > thresh):
            remove.append(i)

# list of outliers
remove = list(set(remove))
remove.sort()

print("Initial Length of dataset ==> ",len(dt))

# Removing outliers from the dataset
for i in range(len(remove) - 1, -1, -1):
    dt = np.delete(dt, remove[i], 0)
print("Final Lenghth of dataset ==> ",len(dt))
print("No. of Outliers removed ==> ",len(remove))

dt = pd.DataFrame(dt)
dt.columns = cols
dataset = dt

# Separating the dataset into features and target variables
df_x = dataset.drop(['Attrition'], axis=1)
df_y = dataset['Attrition']
df_x.shape

# Scaling the features to treat the outliers
mms = MinMaxScaler() # InBuilt Sklearn Scaler Library
dt_x = df_x

# Fitting and Transforming the data
dt_x[["Age", "DailyRate", "DistanceFromHome", "HourlyRate", "MonthlyRate", "MonthlyIncome", "NumCompaniesWorked", "PercentSalaryHike", "TotalWorkingYears", "TrainingTimesLastYear", "YearsAtCompany", "YearsWithCurrManager", "YearsInCurrentRole", "YearsSinceLastPromotion"]] = mms.fit_transform(dt[["Age", "DailyRate", "DistanceFromHome", "HourlyRate", "MonthlyRate", "MonthlyIncome", "NumCompaniesWorked", "PercentSalaryHike", "TotalWorkingYears", "TrainingTimesLastYear", "YearsAtCompany", "YearsWithCurrManager", "YearsInCurrentRole", "YearsSinceLastPromotion"]])
df_x = dt_x

# Initializing Sklearn's PCA object
pca = PCA().fit(df_x)

# Plotting the Cummulative Variance and the number of components
plt.figure(figsize=(30,10))
plt.plot(np.cumsum(pca.explained_variance_ratio_)*100)
plt.xlabel('Number of components')
plt.ylabel('Cummulative explained Variance')
plt.title('PCA')

# Initialization of PCA and transforming the data
pca = PCA(n_components=3)
pca.fit(df_x)
df_x_pca = pca.transform(df_x)

# Initializing the 3D plot
fig = plt.figure(1,figsize=(10,10))
plt.clf()
ax = Axes3D(fig, rect = [0,0,.95,1],elev = 48,azim = 134)
plt.cla()

df_y_temp = df_y.apply(lambda x: 'r' if x == 1 else 'g')

# Plotting the 3 PCA components
for name, label in [('Attrition-Yes',1), ('Attrition-No',0)]:
    ax.text3D(df_x_pca[df_y == label, 0].mean(),
    df_x_pca[df_y == label,1].mean()+1.5,
    df_x_pca[df_y == label ,2].mean(),
    name,horizontalalignment='center',
    bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
ax.scatter(df_x_pca[:,0],df_x_pca[:,1],df_x_pca[:,2], c = df_y_temp,cmap=plt.cm.nipy_spectral,edgecolor='k')
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
plt.title("PCA visualization with 3 components")
plt.show()

def baseline_classifier(model,params, x, y):
    """
    The method converts the input dataframes into numpy arrays of appropriate shapes.
    Then the classifier object is instatiated by input params and model class
    The method then generates scores of parameters(accuracy, precision, recall, f1) using kfold cross validation

    Args: 
      model(): The model class
      params(dict): A dictionary of parameters for the model
      x(Dataframe): Features of the data
      y(Dataframe): Output labels of the data

    Prints the values of the metric in a tabular form
    """
    # Initializing the model
    model = model(**params)

    # Initializing the kfold
    kfold = KFold(n_splits = 10)
    
    # Converting dataframes into numpy arrays
    x = df_x.iloc[:,:].values
    y = df_y.iloc[:].values
    y = np.array([y]).T
    # print("X.shape = "+str(x.shape)+" || Y.shape = "+str(y.shape))
    # print()

    # Generating cross validated scores
    scores = cross_validate(model,x,y,cv = kfold,scoring = ['accuracy','f1','precision','recall'])

    accuracy = round(np.average(scores['test_accuracy'])*100,2)
    precision = round(np.average(scores['test_precision'])*100,2)
    recall = round(np.average(scores['test_recall'])*100,2)
    f1_Score = round(np.average(scores['test_f1'])*100,2)

    # Printing the tabular data
    data = [[accuracy,precision,recall,f1_Score]]
    print (tabulate(data, headers=["Accuracy", "Precision", "Recall", "F1 Score"]))
    # print()
    # print(model)
    # print()
    
def modulated_freq_classifier(model, params, x, y, freq_flag):
    """
    The method converts the input dataframes into numpy arrays of appropriate shapes.
    Then the classifier object is instatiated by input params and model class
    The method then generates scores of parameters(accuracy, precision, recall, f1) using kfold cross validation

    Args: 
      model(): The model class
      params(dict): A dictionary of parameters for the model
      x(Dataframe): Features of the data
      y(Dataframe): Output labels of the data
      freq_flag(string): if 'upsample' then the input data is upsampled using SMOTE
                        if 'downsample' then the input data is downsampled

    Prints the values of the metric in a tabular form
    """

    # Initializing the model
    model = model(**params)

    # Initializing the kfold
    kfold = KFold(n_splits = 10)

    # Converting dataframes into numpy arrays
    x = df_x.iloc[:,:].values
    y = df_y.iloc[:].values
    y = np.array([y]).T
    # print("X.shape = "+str(x.shape)+" || Y.shape = "+str(y.shape))
    # print()

    # initializing metric variables
    accuracy = 0
    precision = 0
    recall = 0
    f1_Score = 0

    # Looping through the folds
    for train_idx, test_idx in kfold.split(x):

        # Generating test and training set
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if freq_flag == "upsample":
            # Initializing SMOTE with a fixed random state for reproducible results
            sm = SMOTE(random_state = 12)

            # Transforming the training data
            x_train_sm, y_train_sm = sm.fit_sample(x_train, np.ravel(y_train, order = 'C'))
            
            # Fitting the transformed data in the model
            model.fit(x_train_sm, y_train_sm)

            # Predicting the model on the test set
            y_test_pred = model.predict(x_test)

            # Storing the metrics
            accuracy = accuracy + accuracy_score(y_test, y_test_pred)
            precision = precision + precision_score(y_test, y_test_pred)
            recall = recall + recall_score(y_test, y_test_pred)
            f1_Score = f1_Score + f1_score(y_test, y_test_pred)


        elif freq_flag == "downsample":
            # Combining both the dataframes
            combo = np.concatenate((x_train,y_train),axis = 1)
            df = pd.DataFrame(data=combo)

            # Separating the dataframe on based of the output labels
            df_majority = df[df.iloc[:,-1] == 0]
            df_minority = df[df.iloc[:,-1] == 1]

            # Downsampling the majority class down to minority class
            df_majority_downsampled = resample(df_majority,replace = False,n_samples = len(df_minority), random_state = 123)

            # Recombining the modified classes
            df_downsampled = pd.concat([df_majority_downsampled,df_minority])
            x_train = df_downsampled.iloc[:,:-1].values
            y_train = df_downsampled.iloc[:,-1].values
            y_train = np.array([y_train]).T

            # Fitting the model on the data
            model.fit(x_train, y_train)

            # Generating the predicted values
            y_test_pred = model.predict(x_test)
            
            # Storing the metrics
            accuracy = accuracy + accuracy_score(y_test, y_test_pred)
            precision = precision + precision_score(y_test, y_test_pred)
            recall = recall + recall_score(y_test, y_test_pred)
            f1_Score = f1_Score + f1_score(y_test, y_test_pred)

    # Rounding off the metrics    
    accuracy = round(accuracy*10,2)
    precision = round(precision*10,2)
    recall = round(recall*10,2)
    f1_Score = round(f1_Score*10,2)

    # Printing the data
    data = [[accuracy,precision,recall,f1_Score]]
    print (tabulate(data, headers=["Accuracy", "Precision", "Recall", "F1 Score"]))
    # print()
    # print(model)
    # print()
    
def returnData(model,params, x, y):
    """
    Returns the evaluation metrics for a particular dataset and params
        The method converts the input dataframes into numpy arrays of appropriate shapes.
    Then the classifier object is instatiated by input params and model class
    The method then generates scores of parameters(accuracy, precision, recall, f1) using kfold cross validation

    Args: 
      model(): The model class
      params(dict): A dictionary of parameters for the model
      x(Dataframe): Features of the data
      y(Dataframe): Output labels of the data

    returns:
      arr : evaluation metrics including accuracy precision and recall.
    """
    model = model(**params)
    kfold = KFold(n_splits = 10)
    
    x = df_x.iloc[:,:].values
    y = df_y.iloc[:].values
    y = np.array([y]).T
    # print("X.shape = "+str(x.shape)+" || Y.shape = "+str(y.shape))
    # print()

    scores = cross_validate(model,x,y,cv = kfold,scoring = ['accuracy','f1','precision','recall'])

    accuracy = round(np.average(scores['test_accuracy'])*100,2)
    precision = round(np.average(scores['test_precision'])*100,2)
    recall = round(np.average(scores['test_recall'])*100,2)
    f1_Score = round(np.average(scores['test_f1'])*100,2)

    data = [accuracy,precision,recall,f1_Score]
    return data 

def makeGraphMD(model, df_x, df_y):
    """
    Plots the graph for tree algorithms of depth vs values of metrics
    Args: 
      model(): The model class
      df_x(Dataframe): Features of the data
      df_y(Dataframe): Output labels of the data
    """
    precision = []
    recall = []
    f1 = []
    vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    for i in vals:
        # print(i)
        data = returnData(model,{'random_state':27, 'max_depth':i},df_x,df_y)
        # print(data)
        precision.append(data[1])
        recall.append(data[2])
        f1.append(data[3])

    plt.figure()
    plt.plot(vals, precision, label = "Precision")
    plt.plot(vals, recall, label = "Recall")
    plt.plot(vals, f1, label = "F1-score")
    plt.xlabel("Depth")
    plt.ylabel("Values")
    plt.legend()
    plt.show()    
    
    
baseline_classifier(LogisticRegression,{},df_x,df_y)
modulated_freq_classifier(LogisticRegression,{},df_x,df_y,"upsample")
modulated_freq_classifier(LogisticRegression,{},df_x,df_y,"downsample")      
baseline_classifier(GaussianNB,{},df_x,df_y)
modulated_freq_classifier(GaussianNB,{},df_x,df_y,"upsample")
modulated_freq_classifier(GaussianNB,{},df_x,df_y,"downsample")
makeGraphMD(DecisionTreeClassifier, df_x, df_y)
baseline_classifier(DecisionTreeClassifier,{'max_depth':8},df_x,df_y)
modulated_freq_classifier(DecisionTreeClassifier,{},df_x,df_y,"upsample")
modulated_freq_classifier(DecisionTreeClassifier,{},df_x,df_y,"downsample")
makeGraphMD(RandomForestClassifier, df_x, df_y)
baseline_classifier(RandomForestClassifier,{'random_state':27, 'max_depth':7},df_x,df_y)
modulated_freq_classifier(RandomForestClassifier,{'random_state':27},df_x,df_y,"upsample")
modulated_freq_classifier(RandomForestClassifier,{'random_state':27},df_x,df_y,"downsample")
baseline_classifier(Perceptron,{},df_x,df_y)
modulated_freq_classifier(Perceptron,{},df_x,df_y,"upsample")
modulated_freq_classifier(Perceptron,{},df_x,df_y,"downsample")
arr = [100]
precision = []
recall = []
f1 = []
for i in range(5):
    # print(i)
    data = returnData(MLPClassifier, {'random_state':27, 'hidden_layer_sizes':arr, 'alpha':0}, df_x, df_y)
    precision.append(data[1])
    recall.append(data[2])
    f1.append(data[3])
    arr.append(100)

plt.figure()
plt.plot([1, 2, 3, 4, 5], precision, label = "Precision")
plt.plot([1, 2, 3, 4, 5], recall, label = "Recall")
plt.plot([1, 2, 3, 4, 5], f1, label = "F1-score")
plt.xlabel("Number of Hidden Layers")
plt.ylabel("Values")
plt.legend()
plt.show()

precision = []
recall = []
f1 = []
vals = [5, 10, 15, 20, 24, 30, 35]
for i in vals:
    # print(i)
    data = returnData(MLPClassifier, {'random_state':27, 'hidden_layer_sizes':[i], 'alpha':0}, df_x, df_y)
    precision.append(data[1])
    recall.append(data[2])
    f1.append(data[3])

plt.figure()
plt.plot(vals, precision, label = "Precision")
plt.plot(vals, recall, label = "Recall")
plt.plot(vals, f1, label = "F1-score")
plt.xlabel("Number of Hidden Units")
plt.ylabel("Values")
plt.legend()
plt.show()

for i in ['relu', 'identity', 'logistic', 'tanh']:
    baseline_classifier(MLPClassifier, {'random_state':27, 'hidden_layer_sizes':[24], 'activation':i}, df_x, df_y)
    
modulated_freq_classifier(MLPClassifier, {'random_state':27, 'hidden_layer_sizes':[24], 'activation':'logistic'}, df_x, df_y, 'upsample')
modulated_freq_classifier(MLPClassifier, {'random_state':27, 'hidden_layer_sizes':[24], 'activation':'logistic'}, df_x, df_y, 'downsample')
for i in ['linear', 'poly', 'rbf', 'sigmoid']:
    baseline_classifier(SVC, {'kernel':i, 'probability':True}, df_x, df_y)
    
baseline_classifier(SVC,{'kernel':'linear','probability':True},df_x,df_y)
modulated_freq_classifier(SVC,{'kernel':'linear'},df_x,df_y,"upsample")
modulated_freq_classifier(SVC,{'kernel':'linear'},df_x,df_y,"downsample")
makeGraphMD(GradientBoostingClassifier, df_x, df_y)
baseline_classifier(GradientBoostingClassifier, {'max_depth':5}, df_x, df_y)
modulated_freq_classifier(GradientBoostingClassifier, {}, df_x, df_y, "downsample")
modulated_freq_classifier(GradientBoostingClassifier, {}, df_x, df_y, "upsample")
makeGraphMD(XGBClassifier, df_x, df_y)
baseline_classifier(XGBClassifier, {'max_depth':5}, df_x, df_y)
modulated_freq_classifier(XGBClassifier, {}, df_x, df_y, "downsample")
modulated_freq_classifier(XGBClassifier, {}, df_x, df_y, "upsample")
x = df_x.iloc[:,:].values
y = df_y.iloc[:].values
y = np.array([y]).T
# print("X.shape = "+str(x.shape)+" || Y.shape = "+str(y.shape))

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state = 123, stratify =y)

lr = LogisticRegression()
lr.fit(x_train,y_train)

gnb = GaussianNB()
gnb.fit(x_train,y_train)

dt = DecisionTreeClassifier(max_depth = 8)
dt.fit(x_train,y_train)

rf = RandomForestClassifier(max_depth = 7)
rf.fit(x_train, y_train)

mlp = MLPClassifier(hidden_layer_sizes=[24], activation = 'logistic')
mlp.fit(x_train,y_train)

svm = SVC(kernel = 'linear',probability = True)
svm.fit(x_train,y_train)

gbc = GradientBoostingClassifier(max_depth = 5)
gbc.fit(x_train, y_train)

xgb = XGBClassifier(max_depth = 5)
xgb.fit(x_train, y_train)
lr_fpr, lr_tpr, thresholds = roc_curve(y_test, lr.predict_proba(x_test)[:,1])
gnb_fpr, gnb_tpr, thresholds = roc_curve(y_test, gnb.predict_proba(x_test)[:,1])
dt_fpr, dt_tpr, thresholds = roc_curve(y_test, dt.predict_proba(x_test)[:,1])
rf_fpr, rf_tpr, thresholds = roc_curve(y_test, rf.predict_proba(x_test)[:,1])
mlp_fpr, mlp_tpr, thresholds = roc_curve(y_test, mlp.predict_proba(x_test)[:,1])
svm_fpr, svm_tpr, thresholds = roc_curve(y_test, svm.predict_proba(x_test)[:,1])
gbc_fpr, gbc_tpr, thresholds = roc_curve(y_test, gbc.predict_proba(x_test)[:,1])
xgb_fpr, xgb_tpr, thresholds = roc_curve(y_test, xgb.predict_proba(x_test)[:,1])

plt.figure(figsize = (10,10))
plt.plot(lr_fpr, lr_tpr, label = 'Logistic Regression' )
plt.plot(gnb_fpr, gnb_tpr, label = 'GaussianNB')
plt.plot(dt_fpr, dt_tpr,label = 'Decision Tree)')
plt.plot(rf_fpr, rf_tpr, label = 'Random Forest' )
plt.plot(mlp_fpr,mlp_tpr, label = 'MLP' )
plt.plot(svm_fpr, svm_tpr, label = 'SVM')
plt.plot(gbc_fpr, gbc_tpr, label = 'Gradient Boosting Classifier')
plt.plot(xgb_fpr, xgb_tpr, label = 'XGBoost Classifier')

plt.plot([0,1], [0,1],label='Base Rate')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Graph')
plt.legend(loc="lower right")
plt.show() 

lr_pre, lr_re, thresholds = precision_recall_curve(y_test, lr.predict_proba(x_test)[:,1])
gnb_pre, gnb_re, thresholds = precision_recall_curve(y_test,gnb.predict_proba(x_test)[:,1])
dt_pre, dt_re, thresholds = precision_recall_curve(y_test,dt.predict_proba(x_test)[:,1])
rf_pre, rf_re, thresholds = precision_recall_curve(y_test,
rf.predict_proba(x_test)[:,1])
mlp_pre, mlp_re, thresholds = precision_recall_curve(y_test,mlp.predict_proba(x_test)[:,1])
svm_pre, svm_re, thresholds = precision_recall_curve(y_test, svm.predict_proba(x_test)[:,1])
gbc_pre, gbc_re, thresholds = precision_recall_curve(y_test, gbc.predict_proba(x_test)[:,1])
xgb_pre, xgb_re, thresholds = precision_recall_curve(y_test, xgb.predict_proba(x_test)[:,1])

plt.figure(figsize = (10,10))
plt.plot(lr_re,lr_pre,label ='Logistic Regression')
plt.plot(gnb_re,gnb_pre,label ='GaussianNB')
plt.plot(dt_re, dt_pre, label = 'Decision Tree')
plt.plot(rf_re, rf_pre, label = 'Random Forest')
plt.plot(mlp_re, mlp_pre, label = 'MLP')
plt.plot(svm_re, svm_pre, label = 'SVM')
plt.plot(gbc_re, gbc_pre, label = 'Gradient Boosting Classifier')
plt.plot(xgb_re, xgb_pre, label = 'XGBoost Classifier')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Graph')
plt.legend(loc="lower right")
plt.show()

x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.2,random_state = 123)
feature_importances = pd.DataFrame(rf.feature_importances_,index = x_train.columns,columns=['importance']).sort_values('importance', ascending=False)
feature_importances = feature_importances.reset_index()[:4]
f, ax = plt.subplots(figsize=(7,5))

sns.set_color_codes("pastel")
sns.barplot(x="importance", y='index', data=feature_importances,label="Total", color="b")       


sse = {}
for k in range(1,10):
    kmeans = KMeans(n_clusters=k, max_iter=1000, init = 'k-means++').fit(df_x, df_y)
    sse[k] = kmeans.inertia_

plt.figure()
plt.plot(list(sse.keys()),list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()

n_clusters = 2
baseline_classifier(KMeans,{'n_clusters':n_clusters},df_x, df_y)

modulated_freq_classifier(KMeans,{'n_clusters':n_clusters},df_x, df_y, "upsample")

modulated_freq_classifier(KMeans,{'n_clusters':n_clusters},df_x, df_y, "downsample")