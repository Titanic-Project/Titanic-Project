#!/usr/bin/env python
# coding: utf-8

import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Test and Train Datasets

# We define the train dataset with pandas library.


# ANALYZING OF SURVIVINGS

#Number of surviveds and not-surviveds
train_dataset['Survived'].value_counts()
train_dataset['Survived'].value_counts().plot(kind='bar', title='Surviving')
plt.xlabel('0= Not-survived  1= Survived')
plt.ylabel('Frequency')
plt.show()

# Number of survivings by sex
train_dataset.groupby('Sex')['Survived'].value_counts()

# Visualization of survivings by sex
train_dataset.groupby('Sex')['Survived'].value_counts().plot(kind='bar', stacked=True, colormap='winter')
plt.show()

# Better visualization of survivings by sex
sex_survived = train_dataset.groupby(['Sex', 'Survived'])
sex_survived.size().unstack().plot(kind='bar', stacked=True, colormap='winter')
plt.ylabel('Frequency')
plt.title('Survivings by sex')
plt.show()

# Usage of size(),unstack() while examining survivings by passenger class
class_survived = train_dataset.groupby(['Pclass', 'Survived'])
# size() - to count number of rows in each grouping
class_survived.size()

# unstack() - to convert results into a more readable format.
class_survived.size().unstack()

# Visualization of survivings by passenger class
class_survived.size().unstack().plot(kind='bar', stacked=True, colormap='autumn')
plt.xlabel('1st = Upper,   2nd = Middle,   3rd = Lower')
plt.ylabel('Frequency')
plt.title('Survivings by passenger class')
plt.show()

# Numbers of survived/not survived passengers by sex and passenger class
print ('Surviving numbers of male passengers by passenger class: ',
male_passenger.groupby(['Pclass', 'Survived']).size().unstack())

print ('Surviving numbers of female passengers by passenger class:',
female_passenger.groupby(['Pclass', 'Survived']).size().unstack())

#Visualization of male and female survivings by passenger class
fig, axes = plt.subplots(nrows=2, ncols=1)
male_passenger.groupby(['Pclass','Survived']).size().unstack().plot(kind='bar', title='Surviving of male passengers by class',
                                                                    stacked=True, colormap='summer', ax=axes[0])
female_passenger.groupby(['Pclass','Survived']).size().unstack().plot(kind='bar', title='Surviving of female passengers by class',
                                                                      stacked=True, colormap='summer', ax=axes[1])
plt.tight_layout()
plt.show()

#Show the missing columns

#List columns with missing values in training set
print(train_dataset.columns[train_dataset.isna().any()])

#Also we can show so the mşssing values
print("Missings in the train data: ")
display(train_dataset.isnull().sum())

print("Missings in the test data: ") 
display(test_dataset.isnull().sum())

#Combining train and test data set with pandas library's concat method

def concat_df(train_dataset,test_dataset):
    return pd.concat([train_dataset, test_dataset], sort= True).reset_index(drop=True)

# We are missing in the age, cabin and beginnings column in the training data.
#Test data set has missing age, wage and cabin column.
# We will merge both data sets and perform data cleaning for the entire data set.
#train ve test 

df_all = concat_df(train_dataset , test_dataset)

#1-Age

#To see how much data our age column is missing as a percentage
print("Missings for Age in the entire data set: " + str(df_all['Age'].isnull().sum()))
print("Missings in percentage: " + str(round(df_all['Age'].isnull().sum()/len(df_all)*100,0)))

# We continue to evaluate it in our age data set
      
print('Median for Age seperated by Pclass: ')    
display(train_dataset.groupby('Pclass')['Age'].median())      
print('Median for Age seperated by Pclass and Sex:')    
display(train_dataset.groupby(['Pclass','Sex'])['Age'].median()) 
print('Number of cases:')    
display(train_dataset.groupby(['Pclass','Sex'])['Age'].count()) 
      
#replace the missings values with the medians of each group
df_all['Age']= df_all.groupby(['Pclass','Sex'])['Age'].apply(lambda x:x.fillna(x.median()))


#2-Fare  

df_all.loc[df_all['Fare'].isnull()]   
      
# We have one missing charge value across the entire data set. Mr. Thomas    
      
#loc cases which are similar to Mr.Thomas and use the median of fare to replace  the missing for his data set
      
mr_thomas=df_all.loc[(df_all['Pclass']==3)&(df_all['SibSp']==0)&(df_all['Embarked']=='S')]['Fare'].median()
print(mr_thomas)
      

#3-Cabin
      
display(train_dataset['Cabin'].unique())
print("There are "+ str(train_dataset['Cabin'].nunique()) + " different values for Cabin and " + str(train_dataset['Cabin'].isnull().sum()) + " cases are missing.")

#keep all first letters of cabin in a new variable and use "M" for each missing
df_all['Deck'] = df_all['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M' )
      
df_all[['Deck','Survived']].groupby('Deck')['Survived'].mean().plot(kind='bar',figsize=(15,7))
plt.suptitle('Survival rates for diffrent cabins')
            
# There are significant differences in survival rates because guests on the upper decks were faster on the lifeboats.
# We will group some decks.      
      
      
idx=df_all[df_all['Deck'] == 'T'].index
df_all.loc[idx,'Deck'] = 'A'   
df_all['Deck']= df_all['Deck'].replace(['A','B','C'],'ABC')
df_all['Deck']= df_all['Deck'].replace(['D','E'],'DE')
df_all['Deck']= df_all['Deck'].replace(['F','G'],'FG')      
df_all['Deck'].value_counts()


#4-Embarked
      
df_all.loc[df_all['Embarked'].isnull()]
           
df_all.loc[(df_all['Pclass'] == 1) & (df_all['Fare'] <= 80) & (df_all ['Deck'] == 'ABC')]['Embarked'].value_counts()

# There are only two missing to start.
#As we have already tried for the fee case, we can look at similar situations to replace the missing value.
      
df_all.loc[df_all['Embarked'].isnull(),'Embarked'] = 'S'
           
print("Missing in the data:")
display(df_all.isnull().sum())

df_all.boxplot(column=['Fare'], figsize=(15,7))

df_all.boxplot(column=['Age'], figsize=(15,7))
      
df_all['Fare'] = pd.qcut(df_all['Fare'], 5 )
df_all['Age'] = pd.cut(df_all['Age'].astype(int), 5 )

print("For age, each category has a different number of cases:")
df_all['Age'].value_counts()
      
print("For fare, each category has a different number of cases:")
df_all['Fare'].value_counts()

df_all[['Age', 'Survived']].groupby('Age')['Survived'].mean()

df_all[['Fare', 'Survived']].groupby('Fare')['Survived'].mean()      

df_all[['Age', 'Survived']].groupby('Age')['Survived'].mean().plot(kind='bar', figsize=(15,7))
plt.suptitle('Survival rates for age categories')      
 
df_all[['Fare', 'Survived']].groupby('Fare')['Survived'].mean().plot(kind='bar', figsize=(15,7))
plt.suptitle('Survival rates for fare categories')         
  
#5-Sibs and Parch

#There are two interesting variables in our dataset that tell us something about family size.
#SibSp defines how many siblings and spouses a passenger has and how many parents and children parch.
# We can sum these variables and add 1 to get the family size (for each passerby).     
      
df_all['Family_Size'] = df_all['SibSp'] + df_all['Parch'] + 1
df_all['Family_Size'].hist(figsize=(15,7))
      
df_all['Family_Size_bin']=df_all['Family_Size'].map(lambda s: 1 if s == 1 else (2 if s == 2 else (3 if 3 <= s <= 4 else (4 if s >= 5 else 0))))

df_all['Family_Size_bin'].value_counts()

#One thesis is that families have a higher chance of survival than singles because they are better able to support themselves and are primarily rescued.
# However, if families are very large, coordination will probably be very difficult in an exceptional situation.

df_all[['Family_Size_bin','Survived']].groupby('Family_Size_bin')['Survived'].mean().plot(kind='bar' , figsize=(15,7))
plt.suptitle('Survival rates for family size categories')

#6-Tickets
df_all['Ticket_Frequency'] = df_all.groupby('Ticket')['Ticket'].transform('count')

#We expect a correlation between ticket frequencies and survival rates because the same ticket numbers are an indication that people travel together.
df_all[['Ticket_Frequency','Survived']].groupby('Ticket_Frequency').mean()

#7-Title

# The name gives us very important information about the socioeconomic status of a traveler.
# We can answer the question of whether someone is married or has an official title that may indicate a higher social status.

df_all['Title'] = df_all['Name'].str.split(',', expand=True)[1].str.split('.',expand=True)[0]
df_all['Is_Married'] = 0
df_all['Is_Married'].loc[df_all['Title'] == 'Mrs'] =1
df_all['Title'].nunique()
# There are many different titles in our dataset.
# We only consider the title with more than 10 cases, we will assign all others to the "misc" category.
title_names = (df_all['Title'].value_counts() < 10)
df_all['Title'] = df_all['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
df_all.groupby('Title')['Title'].count()
                
# We will determine the passengers' surnames.
# We can then see if there are any family members included in both the training and testing data set.
import string

def extract_surname(data):
    families=[]
    for i in range(len(data)):
        name = data.iloc[i]
        if '(' in name:
            name_no_bracket = name.split('(')[0]
        else:
            name_no_bracket = name
        family = name_no_bracket.split(',')[0] 
        title = name_no_bracket.split(',')[1].strip().split(' ')[0]
        for c in string.punctuation:
            family = family.replace(c, '').strip()
        families.append(family)
        
    return families
df_all['Family'] = extract_surname(df_all['Name'])

df_all['Family'].nunique()

# People and women with a master's degree have survived significantly more often and also have larger families on average.
# We assume that if a master or woman is marked as a survivor in the training dataset, the family members in the test dataset will also survive.
      
df_all[['Title','Survived','Family_Size']].groupby('Title').mean()

print('Survival rates grouped by families of women in dataset:')
df_all.loc[(df_all['Sex'] == 'female') & (df_all['Family_Size'] > 1)].groupby('Family')['Survived'].mean().hist(figsize=(12,5))

# Women with a family size of 2 or more often die all or none

master_families = df_all.loc[df_all['Title'] == 'Master']['Family'].tolist()
df_all.loc[df_all['Family'].isin(master_families)].groupby('Family')['Survived'].mean().hist(figsize=(12,5))

# The same applies to families of passengers with captains in their title.

women_rate = df_all.loc[(df_all['Sex'] == 'female') & (df_all['Family_Size'] > 1 )].groupby('Family')['Survived'].mean()
master_rate = df_all.loc[df_all['Family'].isin(master_families)].groupby('Family')['Survived'].mean()

combined_rate = women_rate.append(master_rate)

combined_rate_df = combined_rate.to_frame().reset_index().rename(columns={'Survived' : 'Survival_quota'}).drop_duplicates(subset='Family')

df_all =pd.merge(df_all,combined_rate_df ,how = 'left')
                                                                            
                                                                            
                                                                            
df_all['Survival_quota_NA'] = 1
df_all.loc[df_all['Survival_quota'].isnull(), 'Survival_quota_NA']=0                                                                            
df_all['Survival_quota']=df_all['Survival_quota'].fillna(0)   

#When we examined our dataset, we found that it is not necessary to process some columns.
drop_values = ['Name','Ticket','Cabin'] 
train_dataset = train_dataset.drop(drop_values,axis=1)

# we cleared some missing data.
train_dataset = train_dataset.dropna()
train_dataset.info()

passengers = []
columnss = ['Pclass','Sex','Embarked']  #Pclass:yolcu sınıfı,Embarked:biniş noktası
for columnss in sutunlar:
    passengers.append(pd.get_dummies(train_dataset[columnss])) #We used get_dummies to replace categorized data with a placeholder.

passengers = pd.concat(passengers, axis=1) #We used the concat method to obtain the Dataframe.
print(passengers)

passengers = passengers.drop(['female'], axis=1)
train_dataset = pd.concat((train_dataset,yolcular),axis=1)
train_dataset = train_dataset.drop(['Pclass','Sex','Embarked'],axis=1)

print(train_dataset)

X = train_dataset.values
Y = train_dataset['Survived'].values

X = np.delete(X,1,axis=1)


#test size is %30 and train size is %70
X_train, X_test, y_train, y_test=train_test_split(X,Y,test_size=0.3, random_state=0) 
siniflama = tree.DecisionTreeClassifier(max_depth=5)
siniflama.fit(X_train,y_train)
skor = siniflama.score(X_test,y_test)

print("Score: ",skor)

tahminler = siniflama.predict(X)
as_egitim = accuracy_score(tahminler, Y)

print("Doğruluk tablosu skoru: ", as_egitim)

# We obtained the accuracy score of the application above, but the accuracy score alone cannot be a benchmark for success.
# We used a confusion matrix to look at other criteria (error rate, sensitivity, etc.).
#For this, we used the crosstab () method from the pandas library.

confusion_matrix = pd.crosstab(Y, tahminler, rownames=['Gerçek'], colnames=['Tahmin'])
print (confusion_matrix)

#--------------------------------------------------------   
model = RandomForestClassifier(criterion = 'gini',n_estimators=1750,max_depth=7,min_samples_split =6, min_samples_leaf = 6, max_features = 'auto', oob_score= True, random_state=42, n_jobs=-1,verbose =1)
            
model.fit(X_train, y_train)
predictions = model.predict(X_test)  

score = model.score(X_test, y_test)

print(score)

predictions = siniflama.predict(X)
accuracy = accuracy_score(predictions, Y)

print("Doğruluk tablosu skoru: ", accuracy )

