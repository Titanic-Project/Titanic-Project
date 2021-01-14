#Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#---------------------------------------------------------------------------------------

#Test ve Train Verisetini Tanımlama

#Test verisetini pandas kütüphanesi ile tanımlıyoruz.
train_dataset = pd.read_csv(r'C:\Users\pesen\Desktop\new\train.csv')     #(r'C:\Users\Fatma\Downloads\titanic_data.csv')

#Eğitim verisetini pandas kütüphanesi ile tanımlıyoruz.
test_dataset = pd.read_csv(r'C:\Users\pesen\Desktop\new\test.csv')     #(r'C:\Users\Fatma\Downloads\test_data.csv')

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
test_dataset['Survived'].mean()
#---------------------------------------------------------------------------------------

#Ilk 5 veriyi tablo şeklinde göster
train_dataset.head()
test_dataset.head()
#---------------------------------------------------------------------------------------

#Eksik sütünları göster

#eğitim setinde eksik değerleri olan sütunları listele
print(train_dataset.columns[train_dataset.isna().any()])

#test setinde eksik değerleri olan sütunları listele
print(test_dataset.columns[test_dataset.isna().any()])

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


