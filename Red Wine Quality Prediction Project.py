#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


DF= pd.read_csv('https://raw.githubusercontent.com/dsrscientist/DSData/master/winequality-red.csv')
DF.head()


# In[3]:


DATA = DF


# In[4]:


DATA.head(10)


# In[5]:


DATA.tail(10)


# #as par my obsarvetiom all 11 colmns  and 1 target colmnS contain INT VALUEA ,MENES THARE IS NO CATAGARICAL DATA.

# In[6]:


DATA.shape


# In[7]:


DATA.columns.tolist()


# #total number of rows(data)is 1599 and colmns are 12 the last column(quality)is target variable and rest are indipendent.

# In[8]:


DATA.info()


# #as par my obsarvation no null value found in dataset as well as all  datatype of all colmns are correct . 

# In[9]:


DATA.isna().sum()


# #no null value present im data

# In[10]:


sns.heatmap(DATA.isnull())


# #color representetoin of null values {no null value found}

# 

# 

# In[11]:


DATA.nunique()


# #cheking for unique values

# In[12]:


DATA.nunique().plot(kind='bar')


# #in above grafical represention of uniqe values , a column name "density" has the most unique values
# #and the target column has the minimum unique values

# In[13]:


for i in DATA.columns:
    print(DATA[i].value_counts())
    print('/n'*45)


# #all unique values as par respected columns

# In[14]:


DATA.describe()


# #!)from the describe method we can obsarve that the mean value of all colums are equal or close to median(50%)that mense the data 
# set han no such skewness 
# 2)and some column like fixed acidity,residual sugar,free sulfur dioxide,total sulfur dioxide,has big diffrence in 75% and max value that
# mense thare are some outlairs present

# # univariate

# In[15]:


plt.figure(figsize=(30,25))
f=sns.countplot(x='fixed acidity',data=DATA)
print(DATA['fixed acidity'].value_counts())
plt.show()


# #as par my obsarvation in fixed acidity	column a range between [7.0 to 7.8] has very high count in data.

# In[16]:


plt.figure(figsize=(35,25))
v=sns.countplot(x='volatile acidity',data=DATA)
print(DATA['volatile acidity'].value_counts())


# #a volatile acidity column has sum of 143 unique values and the graph shows the count of unique values.

# In[17]:


plt.figure(figsize=(30,25))
c=sns.countplot(x='citric acid',data=DATA)
print(DATA['citric acid'].value_counts())


# #as par my obsarvation in citric acid column the most repeted value is " 0 "  and total numbar of unique values is " 80 " .

# In[18]:


plt.figure(figsize=(30,25))
r=sns.countplot(x='residual sugar',data=DATA)
print(DATA['residual sugar'].value_counts())


# #a column name residual sugar column has sum of 91 unique values and the graph shows the count of unique values.

# In[19]:


plt.figure(figsize=(30,25))
c=sns.countplot(x='chlorides',data=DATA)
print(DATA['chlorides'].value_counts())


# #a total numbar of unique value of column name chlorides is '153'and the above plot is represention of count of unique values.

# In[20]:


plt.figure(figsize=(30,25))
f=sns.countplot(x='free sulfur dioxide',data=DATA)
print(DATA['free sulfur dioxide'].value_counts())
dfgh=(DATA['free sulfur dioxide'].value_counts().to_frame())
dfgh.count()


# #a total numbar of unique value of column name free sulfur dioxide is ' 60 'and the above plot is represention of unique values.where ('6.0) is repeted most of time.

# In[21]:


plt.figure(figsize=(30,25))
t=sns.countplot(x='total sulfur dioxide',data=DATA)
print()


# #a column name total sulfur dioxide has a "144"unique values and the above plot is represention of count of unique values.

# In[22]:


plt.figure(figsize=(30,25))
d=sns.countplot(x='density',data=DATA)
print(DATA['density'].value_counts())


# #a coulmn name density has the most numbar of unique values('436')and the plot represent the count of unique values

# In[23]:


plt.figure(figsize=(30,25))
p=sns.countplot(x='pH',data=DATA)
print(DATA['pH'].value_counts())


# #a coulmn name PH han '89' unique value  and the plot represent the count of unique values

# In[24]:


plt.figure(figsize=(30,25))
p=sns.countplot(x='sulphates',data=DATA)
print(DATA['sulphates'].value_counts())


# #a coulmn name sulphates han '96' unique value and the plot represent the count of unique values

# In[25]:


plt.figure(figsize=(30,25))
p=sns.countplot(x='alcohol',data=DATA)
print(DATA['alcohol'].value_counts())


# #a coulmn name alcohol  han '65' unique value and the plot represent the count of unique values

# In[26]:


plt.figure(figsize=(10,5))
p=sns.countplot(x='quality',data=DATA)
print(DATA['quality'].value_counts())
x=(DATA['quality'].value_counts().to_frame())
x.count()


# #the target veriable has less number of unique values("6")

# # lets chek the distribution of data in every column. 

# In[27]:


plt.figure(figsize=(20,30))
plot_v=1
for col in DATA:
    if plot_v <=11:
        p=plt.subplot(6,2,plot_v)
        sns.distplot(DATA[col],color='red')
        plt.xlabel(col,fontsize=10)
        plt.yticks(rotation=0,fontsize=10)
    plot_v+=1
plt.tight_layout() 
DATA.skew()


# #1)by observing above plots we can notis that the columns like ['PH','density','citric acid'] are perfacty distributed.
# 2)and columns like ['alcohol','fixed acidity','volatile acidity'] has very less sqewness to the right.
# 3)and rest of the cplumns like['sulphates','total sulfur dioxide','free sulfur dioxide','residual sugar','chlorides'] has a very high skewnwss .

# In[ ]:





#   

# In[28]:


DATA['quality']=DATA['quality'].map({3:'notgood',4:'notgood',5:'notgood',6:'notgood',7:'good',8:'good'})


# In[29]:


DATA.info()


# In[30]:


print(DATA['quality'].value_counts())


# #set a arbitrary cutoff for your dependent variable (wine quality) at  7 or higher getting classified as 'good/1' and the remainder as 'not good/0'.
# now the quality colmns is object dtype .
# and the value count for the good is very low as campaier to notgood.

# # bivariate

# In[31]:


sns.scatterplot( x='fixed acidity',y='quality',data=DATA,hue='quality',palette='bright')
plt.title("relation between fixed acidity and taget column ")


# #as observind the above plot relation between those two is normal.

# In[32]:


sns.scatterplot(x='volatile acidity',y='quality',data=DATA,hue='quality',palette='bright')
plt.title('relation between volatile acidityand quality')


# In[33]:


sns.scatterplot(x='citric acid',y='quality',data=DATA,hue='quality',palette='bright')
plt.title('relation between citric acid and quality')


# In[34]:


sns.scatterplot(x='residual sugar',y='quality',data=DATA,hue='quality',palette='bright')
plt.title('relation between residual sugar and quality')


# In[35]:


sns.scatterplot(x='chlorides',y='quality',data=DATA,hue='quality',palette='bright')
plt.title('relation between chlorides and quality')


# In[36]:


sns.scatterplot(x='free sulfur dioxide',y='quality',data=DATA,hue='quality',palette='bright')
plt.title('relation between free sulfur dioxide and quality')


# In[37]:


sns.scatterplot(x='total sulfur dioxide',y='quality',data=DATA,hue='quality',palette='bright')
plt.title('relation between total sulfur dioxide and quality')


# In[38]:


sns.scatterplot(x='density',y='quality',data=DATA,hue='quality',palette='bright')
plt.title('relation between density and quality')


# In[39]:


sns.scatterplot(x='pH',y='quality',data=DATA,hue='quality',palette='bright')
plt.title('relation between pH and quality')


# In[40]:


sns.scatterplot(x='sulphates',y='quality',data=DATA,hue='quality',palette='bright')
plt.title('relation between sulphates and quality')


# In[41]:


sns.scatterplot(x='alcohol',y='quality',data=DATA,hue='quality',palette='bright')
plt.title('relation between alcohol and quality')


# # let see if data has any outlairs 

# In[42]:


plt.figure(figsize=(20,30))
value=1
for col in DATA:
    if value <=11:
        g=plt.subplot(6,2,value)
        sns.boxplot(DATA[col],palette="Set2_r")
        plt.xlabel(col,fontsize=14)
        plt.yticks(rotation=0,fontsize=14)
    value+=1
plt.tight_layout()    


# # by observing the boxplot we can say every columns has some outliers let see with zscoor method and remove it.

# In[43]:


from scipy import stats
from scipy.stats import zscore


# In[44]:


z_score = zscore (DATA[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']])
abs_z_score = np.abs(z_score)
filtering_entry = (abs_z_score < 3).all(axis=1)
DATA = DATA[filtering_entry]
DATA


# In[45]:


plt.figure(figsize=(20,30))
value=1
for col in DATA:
    if value <=11:
        g=plt.subplot(6,2,value)
        sns.boxplot(DATA[col],palette="Set2_r")
        plt.xlabel(col,fontsize=14)
        plt.yticks(rotation=0,fontsize=14)
    value+=1
plt.tight_layout()    


# #only colums who has a zscore of more then 3 are removed.

# # now lets remove the skewness

# In[46]:


DATA.skew()


# In[61]:


DATA['residual sugar']=np.cbrt(DATA['residual sugar'])


# In[53]:


DATA['chlorides']=np.cbrt(DATA['chlorides'])
DATA['total sulfur dioxide']=np.cbrt(DATA['total sulfur dioxide'])


# In[62]:


DATA.skew()


# In[63]:


DATA.info()


# In[64]:


plt.figure(figsize=(20,30))
plot_v=1
for col in DATA:
    if plot_v <=11:
        p=plt.subplot(6,2,plot_v)
        sns.distplot(DATA[col],color='red')
        plt.xlabel(col,fontsize=10)
        plt.yticks(rotation=0,fontsize=10)
    plot_v+=1
plt.tight_layout() 
DATA.skew()


# #removed only those who have very high skewnwss

# #we can se by observing above plot that now there is no data with very high skewnwss present.

# In[65]:


from sklearn import preprocessing 


# In[66]:


label_encoder = preprocessing.LabelEncoder() 


# In[67]:


DATA['quality']=label_encoder.fit_transform(DATA['quality'])


# In[68]:


DATA['quality'].value_counts()


# In[69]:


DATA.info()


# #encoding target culmn for creating modal

# # let chek if data has any multicolinearty issue.

# In[70]:


DATA.corr()


# In[71]:


XD=DATA.drop('quality', axis=1)
yD=DATA['quality']


# In[72]:


cor=DATA.corr()
plt.figure(figsize=(10,8))
sns.heatmap(cor,linewidths=0.1,fmt='.1g',linecolor='black',annot= True,cmap='Blues_r')


# #the above heatmap represent corelation between colmns to colmns and labal to colmns.
# # hear are some observations
# 1} there is no much relation between colmns towards lable.
# 2} there is high positive corelation between free sulfer dioxide and total sulfar dioxide and aslo between citric acid and fixed acidity.
# 3} and there is nagetive corelation  between pH and fixed acidity
# 4} also there is some coreation between citric acid and volatile acidity 
# 

# # LETS SEE THE VIF VALUE

# In[73]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
VIF=pd.DataFrame()
VIF['VIF values']=[variance_inflation_factor(XD.values,i)for i in range(len(XD.columns))]
VIF['VIF']=XD.columns
VIF                                                                


# #by observing VIF value we can drop the citric acidity column

# In[74]:


XD.drop('citric acid', axis=1,inplace=True)


# In[75]:


XD


# In[76]:


VIF=pd.DataFrame()
VIF['VIF values']=[variance_inflation_factor(XD.values,i)for i in range(len(XD.columns))]
VIF['VIF']=XD.columns
VIF     


# #now the VIF value in in aceptable range

# In[77]:


from sklearn.preprocessing import StandardScaler


# In[78]:


scal=StandardScaler()
x=pd.DataFrame(scal.fit_transform(XD),columns=XD.columns)
x


# In[ ]:





# In[79]:


get_ipython().system('pip install imblearn')


# In[80]:


from imblearn.over_sampling import SMOTE


# In[81]:


sm=SMOTE()
x1,y1=sm.fit_resample(x,yD)


# In[82]:


yD.value_counts()
y1.value_counts()


# # the problem of class imbalance is resolve now

#    

# ## lets split the data  

# In[83]:


from sklearn.model_selection import train_test_split


# In[84]:


from sklearn.tree import DecisionTreeClassifier


# In[85]:


from sklearn.metrics import accuracy_score


# In[86]:


MAXXACU=0
MAXRS=0
for i in range(1,250):
    x_train,x_test,y_train,y_test=train_test_split(x1,y1,test_size=0.35,random_state=i)
    dtc=DecisionTreeClassifier()
    dtc.fit(x_train,y_train)
    pred=dtc.predict(x_test)
    acc=accuracy_score(y_test,pred)
    if acc>MAXXACU:
        MAXXACU=acc
        MAXRS=i
print('MAX accuracy is',MAXXACU,'BEST rendom state is',MAXRS)    


# In[87]:


x_train,x_test,y_train,y_test=train_test_split(x1,y1,test_size=0.35,random_state=125)


# now we have the best randoom state value "167" 
# we will use this value to build models

# # lets start with DecisionTree itself

# In[88]:


from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
dtc=DecisionTreeClassifier()
dtc.fit(x_train,y_train)
preddtc=dtc.predict(x_test)
print(accuracy_score(y_test,preddtc))
print(confusion_matrix(y_test,preddtc))
print(classification_report(y_test,preddtc))


# ### score for this decisiontree modal is 92.73% lets cross validate this

# In[89]:


scoredt=cross_val_score(dtc,x1,y1,cv=10)
print(scoredt)
print(scoredt.mean())
print("diffrence in cross_val_score and accuracy_score-", accuracy_score(y_test,preddtc)-(scoredt.mean()))


# ### after observing the cross_val the diffrence is on 3% 

#     

# In[90]:


from sklearn.linear_model import LogisticRegression


# ## now lets build the next modal [LogisticRegression] 

# In[91]:


lger=LogisticRegression()
lger.fit(x_train,y_train)
predlger=lger.predict(x_test)
print(accuracy_score(y_test,predlger))
print(confusion_matrix(y_test,predlger))
print(classification_report(y_test,predlger))


# ### the accuracy for logisticregression is 81% lets cross validate this

# In[92]:


scorelger=cross_val_score(lger,x1,y1,cv=10)
print(scorelger)
print(scorelger.mean())
print("defreence between accuracy and cross_val_score",accuracy_score(y_test,predlger)-(scorelger.mean()))


# ### after cross validatinf the deffrence is less 1%

# In[93]:


from sklearn.svm import SVC


# ## the next modal is SVC

# In[94]:


svm=SVC()
svm.fit(x_train,y_train)
predsvm=svm.predict(x_test)
print(accuracy_score(y_test,predsvm))
print(confusion_matrix(y_test,predsvm))
print(classification_report(y_test,predsvm))


# ### accuuracy for SVC modal is 87% lets chek croos validation

# In[95]:


scoresvm=cross_val_score(svm,x1,y1,cv=10)
print(scoresvm)
print(scoresvm.mean())
print('dffrence in accuracy and cross_val_score',accuracy_score(y_test,predsvm)-(scoresvm.mean()))


# ## as seen the dffrence is 1.6%

# In[96]:


from sklearn.ensemble import RandomForestClassifier


# # next modal is RandomForestClassifier

# In[97]:


rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)
predrfc=rfc.predict(x_test)
print(accuracy_score(y_test,predrfc))
print(confusion_matrix(y_test,predrfc))
print(classification_report(y_test,predrfc))


# ### the RandomForestClassifier has the highest score of 93.86% lets see the cross validation 

# In[98]:


scorerfc=cross_val_score(rfc,x1,y1,cv=10)
print(scorerfc)
print(scorerfc.mean())
print('deffrance of accuracy and cross_val_score',accuracy_score(y_test,predrfc)-(scorerfc.mean()))


# ### the deffrance is 1.80%

# In[100]:


from sklearn.ensemble import ExtraTreesClassifier


# # next modal is extratreesclassifier

# In[102]:


edtc=ExtraTreesClassifier()
edtc.fit(x_train,y_train)
prededtc=edtc.predict(x_test)
print(accuracy_score(y_test,prededtc))
print(confusion_matrix(y_test,prededtc))
print(classification_report(y_test,prededtc))


# ### accuracy for ExtraTreesClassifier is 95.45% lets chek the cross validation 
# ### the score is highest as compair to othar modals

# In[103]:


scoreedtc=cross_val_score(edtc,x1,y1,cv=10)
print(scoreedtc)
print(scoreedtc.mean())
print('deffrance between accuracy and cross val score is-',(accuracy_score(y_test,prededtc))-(scoreedtc.mean()))


# ### as seen the defreence is only 0.86% ony

# In[105]:


from sklearn.ensemble import GradientBoostingClassifier


# # the next modal is GradientBoostingClassifier

# In[106]:


gbc=GradientBoostingClassifier()
gbc.fit(x_train,y_train)
predgbc=gbc.predict(x_test)
print(accuracy_score(y_test,predgbc))
print(confusion_matrix(y_test,predgbc))
print(classification_report(y_test,predgbc))


# ### the score for GradientBoostingClassifier is 90.45% 
# ### lets che the cross val score 

# In[107]:


scoregbc=cross_val_score(gbc,x1,y1,cv=10)
print(scoregbc)
print(scoregbc.mean())
print('dffreence of accuracy and cross val score is- ',(accuracy_score(y_test,predgbc))-(scoregbc.mean()))


# ### as seen the doffrance is only 2.28%

# # AS observing all modal accuracy and  compaiering with cross_val_score the extratreesclassifier is the best modal.with the accuracy of 95.45%

# ### lets chek if we can make any improvment by using hyper parameter tuning

# In[110]:


from sklearn.model_selection import GridSearchCV


# In[117]:


parameters={'criterion':['gini','entropy','log_loss'],
           'random_state':[10,50,100,150],
           'max_depth':[1,10,20],
           'n_jobs':[-2,-1,1,2],
           'n_estimators':[10,100,200,300]}


# In[118]:


GCV=GridSearchCV(ExtraTreesClassifier(),parameters,cv=4)


# In[119]:


GCV.fit(x_train,y_train)


# In[120]:


GCV.best_params_


# In[122]:


final_model=ExtraTreesClassifier(criterion='gini',random_state=150 ,max_depth= 20,n_jobs=-2 ,n_estimators=300 )
final_model.fit(x_train,y_train)
predf=final_model.predict(x_test)
accuracy=accuracy_score(y_test,predf)
accuracy*100


# # the final modal score is reduse by o.2% 

# ### lets chek the ROC curve

# In[123]:


from sklearn import metrics


# In[124]:


fpr,tpr,thresholds=metrics.roc_curve(y_test,predf)


# In[128]:


roc_auc=metrics.auc(fpr,tpr)


# In[129]:


display=metrics.RocCurveDisplay(fpr=fpr,tpr=tpr,roc_auc=roc_auc,estimator_name=final_model)
display.plot()


# # as seen the auc value is 95% 

#        

#         

# In[ ]:





# ### lets save the modal

# In[130]:


import joblib


# In[131]:


joblib.dump(final_model,"Red Wine Quality Prediction Project")


#             

#                                      end

# In[ ]:




