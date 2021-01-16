#Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

#---------------------------------------------------------------------------------------

#Test ve Train Verisetini Tanımlama

#Test verisetini pandas kütüphanesi ile tanımlıyoruz.
train_dataset = pd.read_csv(r'C:\Users\Fatma\Desktop\train.csv')     #(r'C:\Users\Fatma\Downloads\titanic_data.csv')

#Eğitim verisetini pandas kütüphanesi ile tanımlıyoruz.
test_dataset = pd.read_csv(r'C:\Users\Fatma\Desktop\test.csv')     #(r'C:\Users\Fatma\Downloads\test_data.csv')

#---------------------------------------------------------------------------------------

train_dataset.info()
test_dataset.info()
#---------------------------------------------------------------------------------------

#Sütunları kontrol ediyoruz.
print(train_dataset.columns )
print(test_dataset.columns)

#Sütun ve Satırları Train ve Test için yazdır
print('\ntrain dataset: %s, test dataset %s' %(str(train_dataset.shape), str(test_dataset.shape)) )

#---------------------------------------------------------------------------------------

#İki veri setindeki toplam yolcu sayısı
print(train_dataset.shape[0] + test_dataset.shape[0])

#---------------------------------------------------------------------------------------

#Hayatta kalma oranı
train_dataset['Survived'].mean()

#---------------------------------------------------------------------------------------

#Ilk 5 veriyi tablo şeklinde göster
train_dataset.head()

#---------------------------------------------------------------------------------------

#Eksik sütünları göster

#eğitim setinde eksik değerleri olan sütunları listele
print(train_dataset.columns[train_dataset.isna().any()])

#test setinde eksik değerleri olan sütunları listele
#print(test_dataset.columns[test_dataset.isna().any()])

#Eksiklik verileri şu şekilde de gösterebiliriz
print("Missings in the train data: ")
display(train_dataset.isnull().sum())

print("Missings in the test data: ") 
display(test_dataset.isnull().sum())

#---------------------------------------------------------------------------------------

#train ve test data setini pandas kütüphanesinin concat metoduyla birleştiren fonksyon

def concat_df(train_dataset,test_dataset):
    return pd.concat([train_dataset, test_dataset], sort= True).reset_index(drop=True)
  
#---------------------------------------------------------------------------------------  

#Eğitim verilerinde yaş, kabin ve başlangıçlar sütununda eksiklerimiz var.
#Test veri setinde yaş, ücret ve kabin sütununda eksiklikler var. 
#Her iki veri setini birleştireceğiz ve tüm veri seti için veri temizliğini gerçekleştireceğiz.
#train ve test 

df_all = concat_df(train_dataset , test_dataset)

#---------------------------------------------------------------------------------------

#yaş sütünumuzun yüzdelik olarak ne kadar eksik verisi olduğunu görmek için
print("Missings for Age in the entire data set: " + str(df_all['Age'].isnull().sum()))
print("Missings in percentage: " + str(round(df_all['Age'].isnull().sum()/len(df_all)*100,0)))

#yaş veri setimizin içinde değerlendirmeye devam ediyoruz
      
print('Median for Age seperated by Pclass: ')    
display(train_dataset.groupby('Pclass')['Age'].median())      
print('Median for Age seperated by Pclass and Sex:')    
display(train_dataset.groupby(['Pclass','Sex'])['Age'].median()) 
print('Number of cases:')    
display(train_dataset.groupby(['Pclass','Sex'])['Age'].count()) 
      
#replace the missings values with the medians of each group
df_all['Age']= df_all.groupby(['Pclass','Sex'])['Age'].apply(lambda x:x.fillna(x.median()))

      
#----------------------------------------------------------------------------------------------

#ücretlere bakıyoruz şimdi de      
df_all.loc[df_all['Fare'].isnull()]   
      
#Tüm veri setinde tek bir eksik ücret değerimiz var. Bay Thomas
      
#----------------------------------------------------------------------------------------------      
      
#loc cases which are similar to Mr.Thomas and use the median of fare to replace  the missing for his data set
      
mr_thomas=df_all.loc[(df_all['Pclass']==3)&(df_all['SibSp']==0)&(df_all['Embarked']=='S')]['Fare'].median()
print(mr_thomas)
      

#------------------------------------------------------------------------------------------------------------
#kabine bakıyoruz belki de önemli bir belirleyici olabilir.
      
display(train_dataset['Cabin'].unique())
print("There are "+ str(train_dataset['Cabin'].nunique()) + " different values for Cabin and " + str(train_dataset['Cabin'].isnull().sum()) + " cases are missing.")
      
#------------------------------------------------------------------------------------------------------------
      
#keep all first letters of cabin in a new variable and use "M" for each missing
df_all['Deck'] = df_all['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M' )
      
df_all[['Deck','Survived']].groupby('Deck')['Survived'].mean().plot(kind='bar',figsize=(15,7))
plt.suptitle('Survival rates for diffrent cabins')
      
#------------------------------------------------------------------------------------------------------------      
      
#Hayatta kalma oranlarında önemli farklılıklar var çünkü üst güvertelerdeki konuklar filikalarda daha hızlıydı.
#Bazı desteleri gruplayacağız.      
      
      
idx=df_all[df_all['Deck'] == 'T'].index
df_all.loc[idx,'Deck'] = 'A'   
df_all['Deck']= df_all['Deck'].replace(['A','B','C'],'ABC')
df_all['Deck']= df_all['Deck'].replace(['D','E'],'DE')
df_all['Deck']= df_all['Deck'].replace(['F','G'],'FG')      
df_all['Deck'].value_counts()
      

#------------------------------------------------------------------------------------------------------
      
#gemiye bindiler
      
df_all.loc[df_all['Embarked'].isnull()]
     
#------------------------------------------------------------------------------------------------------
      
df_all.loc[(df_all['Pclass'] == 1) & (df_all['Fare'] <= 80) & (df_all ['Deck'] == 'ABC')]['Embarked'].value_counts()

#Başlamak için sadece iki eksik var.
#Ücret davası için zaten denediğimiz gibi, eksik değeri değiştirmek için benzer durumlara bakabiliriz.
      
df_all.loc[df_all['Embarked'].isnull(),'Embarked'] = 'S'
     
#------------------------------------------------------------------------------------------------------
      
print("Missing in the data:")
display(df_all.isnull().sum())
      
#------------------------------------------------------------------------------------------------------
      
df_all.boxplot(column=['Fare'], figsize=(15,7))

df_all.boxplot(column=['Age'], figsize=(15,7))
      
#------------------------------------------------------------------------------------------------------
      
df_all['Fare'] = pd.qcut(df_all['Fare'], 5 )
df_all['Age'] = pd.cut(df_all['Age'].astype(int), 5 )

print("For age, each category has a different number of cases:")
df_all['Age'].value_counts()
      
print("For fare, each category has a different number of cases:")
df_all['Fare'].value_counts()
#------------------------------------------------------------------------------------------------------

df_all[['Age', 'Survived']].groupby('Age')['Survived'].mean()

df_all[['Fare', 'Survived']].groupby('Fare')['Survived'].mean()      
#------------------------------------------------------------------------------------------------------,

df_all[['Age', 'Survived']].groupby('Age')['Survived'].mean().plot(kind='bar', figsize=(15,7))
plt.suptitle('Survival rates for age categories')      
 
df_all[['Fare', 'Survived']].groupby('Fare')['Survived'].mean().plot(kind='bar', figsize=(15,7))
plt.suptitle('Survival rates for fare categories')         
      
#-----------------------------------------------------------------------------------------------------------------
#Veri setimizde bize aile büyüklüğü hakkında bir şeyler söyleyen iki ilginç değişken var. 
#SibSp, bir yolcunun kaç tane kardeşi ve eşi olduğunu tanımlar ve kaç tane ebeveyn ve çocuk parch. 
#Bu değişkenleri özetleyebilir ve aile boyutunu elde etmek için 1 ekleyebiliriz (her yoldan geçen için).     
      
df_all['Family_Size'] = df_all['SibSp'] + df_all['Parch'] + 1
df_all['Family_Size'].hist(figsize=(15,7))
      
df_all['Family_Size_bin']=df_all['Family_Size'].map(lambda s: 1 if s == 1 else (2 if s == 2 else (3 if 3 <= s <= 4 else (4 if s >= 5 else 0))))

df_all['Family_Size_bin'].value_counts()

#Bir tez, ailelerin hayatta kalma şansının bekarlara göre daha yüksek olduğu, çünkü kendilerini daha iyi destekleyebildikleri ve öncelikli olarak kurtarıldıklarıdır.
# Bununla birlikte, aileler çok büyükse, istisnai bir durumda koordinasyon muhtemelen çok zor olacaktır.

df_all[['Family_Size_bin','Survived']].groupby('Family_Size_bin')['Survived'].mean().plot(kind='bar' , figsize=(15,7))
plt.suptitle('Survival rates for family size categories')

#--------------------------------------------------------------------------------------------------------------------------
#biletler
df_all['Ticket_Frequency'] = df_all.groupby('Ticket')['Ticket'].transform('count')

#Bilet sıklıkları ile hayatta kalma oranları arasında bir korelasyon bekliyoruz, çünkü aynı bilet numaraları, insanların birlikte seyahat ettiklerinin bir göstergesi.
df_all[['Ticket_Frequency','Survived']].groupby('Ticket_Frequency').mean()
#--------------------------------------------------------------------------------------------------------------------------------------
#İsim bize bir yolcunun sosyoekonomik durumu hakkında çok önemli bilgiler veriyor. 
#Birinin evli olup olmadığı veya daha yüksek bir sosyal statünün göstergesi olabilecek resmi bir unvanı olup olmadığı sorusuna cevap verebiliriz.

df_all['Title'] = df_all['Name'].str.split(',', expand=True)[1].str.split('.',expand=True)[0]
df_all['Is_Married'] = 0
df_all['Is_Married'].loc[df_all['Title'] == 'Mrs'] =1
df_all['Title'].nunique()
#Veri setimizde pek çok farklı başlık var.
# Yalnızca 10'dan fazla davaya sahip başlığı dikkate alıyoruz, diğerlerinin tümünü "misc" kategorisine atayacağız.
title_names = (df_all['Title'].value_counts() < 10)
df_all['Title'] = df_all['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
df_all.groupby('Title')['Title'].count()

#----------------------------------------------------------------
#Yolcuların soyadlarını belirleyeceğiz. 
#Daha sonra hem eğitimde hem de test veri setinde bulunan herhangi bir aile üyesi olup olmadığını görebiliriz.
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

#Yüksek lisans derecesine sahip kişiler ve kadınlar, önemli ölçüde daha sık hayatta kaldılar ve aynı zamanda ortalama olarak daha büyük ailelere sahipler. 
#Eğitim veri setinde bir usta veya kadın hayatta kalan olarak işaretlenirse, test veri setindeki aile üyelerinin de hayatta kalacağını varsayıyoruz.
      
df_all[['Title','Survived','Family_Size']].groupby('Title').mean()

print('Survival rates grouped by families of women in dataset:')
df_all.loc[(df_all['Sex'] == 'female') & (df_all['Family_Size'] > 1)].groupby('Family')['Survived'].mean().hist(figsize=(12,5))

#Aile büyüklüğü 2 veya daha fazla olan kadınlarda, çoğu zaman hepsi veya hiçbiri ölmez.

master_families = df_all.loc[df_all['Title'] == 'Master']['Family'].tolist()
df_all.loc[df_all['Family'].isin(master_families)].groupby('Family')['Survived'].mean().hist(figsize=(12,5))

#Aynısı, unvanında kaptan olan yolcu aileleri için de geçerlidir.

women_rate = df_all.loc[(df_all['Sex'] == 'female') & (df_all['Family_Size'] > 1 )].groupby('Family')['Survived'].mean()
master_rate = df_all.loc[df_all['Family'].isin(master_families)].groupby('Family')['Survived'].mean()

combined_rate = women_rate.append(master_rate)

combined_rate_df = combined_rate.to_frame().reset_index().rename(columns={'Survived' : 'Survival_quota'}).drop_duplicates(subset='Family')

df_all =pd.merge(df_all,combined_rate_df ,how = 'left')
                                                                            
                                                                            
                                                                            
df_all['Survival_quota_NA'] = 1
df_all.loc[df_all['Survival_quota'].isnull(), 'Survival_quota_NA']=0                                                                            
df_all['Survival_quota']=df_all['Survival_quota'].fillna(0)                                                                            
                                                                            
                                                                            
#Etiket ve Bir Sıcak Kodlama

non_numeric_features = ['Embarked', 'Sex', 'Title', 'Age', 'Fare', 'Deck']
                                                                            
                                                                            
for feature in non_numeric_features:
    df_all[feature] = LabelEncoder().fit_transform(df_all[feature])
                                                                            
cat_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'Deck', 'Family_Size_bin', 'Age', 'Fare']
                                                                            
encoded_features = []
                                                                            
for feature in cat_features:
    encoded_feat = OneHotEncoder().fit_transform(df_all[feature].values.reshape(-1,1)).toarray()
    n = df_all[feature].nunique()
    cols = ['{}_{}.format(feature,n) for n in range(1, n+1)] 
    encoded_df = pd.DataFrame(encoded_feat, columns= cols)
    encoded_features.append(encoded_df)        
                                                                            
df_all  = pd.concat([df_all, encoded_features], axis=1)     
            
train_dataset, test_dataset = divide_df(df_all)            
                                                                                
                                                                                
                                                                            
#Modelleme ve tahmin
            
            
drop_cols = ['Emabarked', 'Family','Family_Size','Survived','Family_Size_bin','Deck','Age','Name','Parch','PassengerId','Pclass','Sex','SibSp','Title','Ticket','Cabin'] 
            
drop_cols_2 = ['Emabarked', 'Family','Family_Size','Survived','Family_Size_bin','Deck','Fare','Name','Parch','PassengerId','Pclass','Sex','SibSp','Title','Ticket','Cabin']            
      
X = StandardScaler().fit_transform(train_dataset.drop(columns=drop_cols))
y = train_dataset['Survived'].values
            
X_test = StandardScaler().fit_transform(test_dataset.drop(columns=drop_cols2))       
X_train, X_test,y_train,y_test = train_test_split(X,y, test_size = 0.25, random_state =42) 
model = RandomForestClassifier(criterion = 'gini',n_estimators=1750,max_depth=7,min_samples_split =6, min_samples_leaf = 6, max_features = 'auto', oob_score= True, random_state=42, n_jobs=-1,verbose =1)
            
model.fit(X_train, y_train)
predictions = model.predict(X_test)  
print(model.score(X_test, y_test)) 
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived':predictions})
output['Survived'] = output['Survived'].astype(int) 
output.to_cs('2020_04_09_bd_final_v3.csv', index= False)            
            
            
