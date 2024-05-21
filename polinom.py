# Kütüphanelerin Yüklenmesi 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
  
# Çeşitli şişe numunelerindeki verileri içeren veri setini yükleyelim. 
from google.colab import files
uploaded = files.upload()

datas = pd.read_csv("bottle.csv")
datas.head()

# 'T_degC' (Sicaklik),'Salnty' (Tuzluluk) sutunlarini ayiklayalim.
datas_df = datas[['T_degC','Salnty']]

# Sutunları yeniden isimlendirelim.
datas_df.columns = ['Sicaklik', 'Tuzluluk']


# Verileri inceleyelim
import seaborn as sns
sns.pairplot(datas_df, kind="reg")

datas_df.shape

# Null veri var mı?
datas_df.isnull().sum()

# Null (NaN) verilerin olduğu satırları düşürelim.

datas_df.fillna(method='ffill', inplace=True)
datas_df.isnull().sum()

# Tuzluluk sütununu X, Sicaklik sütununu Y bileşenleri olarak ayıralım.
X = np.array(datas_df['Tuzluluk']).reshape(-1, 1)
y = np.array(datas_df['Sicaklik']).reshape(-1, 1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# Bu veriseti üzerinde Doğrusal Regresyon uygulayalım. 
from sklearn.linear_model import LinearRegression 
dogrusal_reg = LinearRegression() 
  
dogrusal_reg.fit(X_train, y_train) 

y_pred = dogrusal_reg.predict(X_test)                                     
dogruluk_puani = dogrusal_reg.score(X_test, y_test)                       
print("Dogrusal Regresyon Modeli Dogruluk Puani: " + "{:.1%}".format(dogruluk_puani))

# Doğrusal Regresyon modelini grafiğe dökelim.

plt.scatter(X_test, y_test, color='r')
plt.plot(X_test, y_pred, color='g')
plt.show()

Doğruluk tayini sonucu iyi çıkmasına rağmen, veri setinin trendini bir lineer fonksiyonun temsil etmesini tercih etmeyebiliriz. Grafikte de görüleceği üzere, veri setinin trendi daha çok bir eğriyi çağrıştırmaktadır. 

# Veriseti üzerinde 4. dereceden Polinom Regresyon uygulayalım. 
# PolynomialFeatures fonksiyonunu çağıralım. 
from sklearn.preprocessing import PolynomialFeatures 

# PolynomialFeatures fonksiyonu, regresyonda kullanılacak eğitim veri setini belirtilen derecede 
# bir polinom olarak algılamak için kullanılan bir ön işleme fonksiyonudur.

poli_reg = PolynomialFeatures(degree = 4) # polinom fonksiyonu tanımlanır.
transform_poli = poli_reg.fit_transform(X_train) # X eğitim verileri bu polinoma uydurulur ve dönüştürülür. 
  

dogrusal_reg2 = LinearRegression() # Şimdi, lineer regresyon fonksiyonumuzu çağırıyoruz. 
dogrusal_reg2.fit(transform_poli,y_train) # Bu fonksiyon, polinoma dönüştürülmüş X eğitim verisi ve y eğitim verisi ile uyumlandırılır.

poli_tahmin = dogrusal_reg2.predict(transform_poli) # polinoma dönüştürülmüş X eğitim veri seti üzerine regresyon fonksiyonu ile tahmin gerçekleştirilir.


#polinom_egitim_dogruluk_puani = dogrusal_reg2.score(X_test, y_test)                 
#print("Polinom Regresyon Modeli Dogruluk Puani: " + "{:.1%}".format(polinom_egitim_dogruluk_puani))
from sklearn.metrics import mean_squared_error,r2_score
rmse = np.sqrt(mean_squared_error(y_train,poli_tahmin))
r2 = r2_score(y_train,poli_tahmin)
print("Test verisi için Kök Karesel Ortalama Hata: " +"{:.2}".format(rmse))
print("Test verisi için R2 Skoru: " +"{:.2}".format(r2))


# Eğitim veri seti üzerine tahmini görselleştirelim.
plt.scatter(X_train, y_train)

import operator
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(X_train,poli_tahmin), key=sort_axis)
X_train, poli_tahmin = zip(*sorted_zip)
plt.plot(X_train, poli_tahmin, color='r', label = 'Polinom Regresyon')
plt.plot(X_test, y_pred, color='g', label = 'Lineer Regresyon')
plt.xlabel('Tuzluluk') 
plt.ylabel('Sıcaklık') 
plt.legend()
plt.show()


Doğruluk Oranı Ve Grafik : https://prnt.sc/-FIGSOWfGwph

