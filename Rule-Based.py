###########################################################
#
# 29/06/2023
# Case Study - Kural Tabanlı Sınıflandırma ile Potansiyel Müşteri Getirisi Hesaplama
#
##########################################################


# Değişken Bilgileri:
# - PRICE – Müşterinin harcama tutarı
# - SOURCE – Müşterinin bağlandığı cihaz türü
# - SEX – Müşterinin cinsiyeti
# - COUNTRY – Müşterinin ülkesi
# - AGE – Müşterinin yaşı



import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option("display.width",1000)
pd.set_option("display.max_columns",None)
plt.matplotlib.use('Qt5Agg')

####################################################################
# Görev 1
####################################################################

### Soru 1: persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.
df = pd.read_csv("2. Hafta/2.Python ile Veri Analizi - Pandas/Datasets/persona.csv")
# kolaylık olsun diye col isimlerini küçülteceğim.
df.columns = [col.lower() for col in df.columns]


def check_df(dataframe, head):
    print("======================= STRUCTURAL INFO =======================")
    print(dataframe.info())
    print("======================= HEAD =======================")
    print(dataframe.head(head))
    print("======================= TAIL =======================")
    print(dataframe.tail(head))
    print("======================= SHAPE =======================")
    print(dataframe.shape)
    print("======================= NDIM =======================")
    print(dataframe.ndim)
    print("======================= TYPES =======================")
    print(dataframe.dtypes)
    print("========================== QUANTILES ========================")
    print(df.describe([0.05, 0.50, 0.95, 0.99]).T)

check_df(df, 5)

# Not: cat_col gibi durumları srouda "genel bilgi" dediği için bakmadım. İleride
# istemezlerse buraya dönüp bakıcam.


### Soru 2: Kaç unique soruce vardır? Frekansları nedir?
# Kaç unique değer için .nunique() fonksiyonunu kulanabiliriz.
df["source"].nunique() # 2 tane
# Uniqeu değerlerin frekansları (aslında sınıfların freakansları demek istemiş):
df["source"].value_counts() # android  2974, ios  2026
# ayrıca bu yolla da kaç adet unique değer olup plmadığını görebilirdin.


### Soru 3: Kaç unique PRICE vardır?
df["price"].nunique() # 6


### Soru 4: Hangi PRICE'dan kaçar tane satış gerçekleşmiş?
df["price"].value_counts()


### Soru 5: Hangi ülkeden kaçar tane satış olmuş?
# 1. yol: Ülkeler değişken olduğu için sınıf değerlerine bakarak yapabiliriz.
df["country"].value_counts()
# 2. yol: Groupby
# buraaki sorun price ve countyr'de null olmamalı
df.groupby("country").agg({"price": "count"})
df.isnull().sum()

### Soru 6: Ülkelere göre satışlardan toplam ne kadar kazanılmış?
df.groupby("country").agg({"price": ["count","sum"]})


### Soru 7: SOURCE türlerine göre satış sayıları nedir?
df.groupby("source").agg({"price": ["count","sum"]})


### Soru 9: SOURCE'lara göre PRICE ortalamaları nedir?
df.groupby("country").agg({"price":["count","sum","mean"]})


### Soru 10: COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?
df.groupby(["country","source"]).agg({"price": "mean"})


####################################################################
# Görev 2
####################################################################

### COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?
df.groupby(["country","source","sex","age"]).agg({"price": "mean"})




####################################################################
# Görev 3:Çıktıyı PRICE’a göre sıralayınız.
####################################################################

### Soru 1: Önceki sorudaki çıktıyı daha iyi görebilmek için sort_values metodunu azalan olacak şekilde PRICE’a göre uygulayınız.
df.groupby(["country","source","sex","age"]).agg({"price": "mean"}).sort_values(ascending=False, by="price")


### Soru 2: Çıktıyı agg_df olarak kaydediniz.
agg_df = df.groupby(["country","source","sex","age"]).agg({"price": "mean"}).sort_values(ascending=False, by="price")
agg_df



####################################################################
# Görev 4: Indekste yer alan isimleri değişken ismine çeviriniz.
####################################################################

### Üçüncü sorunun çıktısında yer alan PRICE dışındaki tüm değişkenler index isimleridir. Bu isimleri değişken isimlerine çeviriniz.
# reset index, içerisindeki değeri değişken haline getirirken diğer bir yanda indexi de 0 lar.
agg_df.reset_index(inplace=True)
agg_df


# şunları bir unut
df2 = df.copy()
df2.columns

df2.index = df2.source + "_" + df2.sex + "_" + df2.country + "_" + str(df2.age)
df2.head()



####################################################################
# Görev 5: Age değişkenini kategorik değişkene çeviriniz ve agg_df’e ekleyiniz.
####################################################################

### Age sayısal değişkenini kategorik değişkene çeviriniz.
### Aralıkları ikna edici şekilde oluşturunuz.
### Örneğin: ‘0_18', ‘19_23', '24_30', '31_40', '41_70'

# labellamak zaten kategorik değişkene çevirmek oluyor.
agg_df["age"]
agg_df["age_cat"] = pd.cut(agg_df['age'], bins=[0, 18, 23, 30, 40, 70], labels=["kid","young","adult","mature","old"],right=False)
agg_df

## Aklıma gele başka bir şeyi deniyorm.
df["age"] = df["age"].astype("category")
pd.cut(df['age'], bins=["0", "18", "23", "30", "40", "70"], labels=["kid","young","adult","mature","old"],right=False)

####################################################################
# Görev 6: Yeni seviye tabanlı müşterileri (persona) tanımlayınız.
####################################################################

# - Yeni seviye tabanlı müşterileri (persona) tanımlayınız ve veri setine değişken olarak ekleyiniz.
# - Yeni eklenecek değişkenin adı: customers_level_based
# - Önceki soruda elde edeceğiniz çıktıdaki gözlemleri bir araya getirerek customers_level_based değişkenini oluşturmanız gerekmektedir.

agg_df["customers_level_based"] = [value[0].upper() + "_" + value[1].upper() + "_" + value[2].upper() +
                                   "_" + value[5].upper() for value in agg_df.values]
agg_df

# Bir de şu yol var
(aggdf['COUNTRY'].str.upper() + "" + aggdf['SOURCE'].str.upper() + ""
                                  + aggdf['SEX'].str.upper() + "" + agg_df['AGE'].str.upper()) # hata alırsan tiplerini aynı şekle getir.



# aynı mnatıkta bir başka

aggDataFrame["customers_level_based"] = (aggDataFrame["COUNTRY"].astype(str) + "" +
                                        aggDataFrame["SOURCE"].astype(str) + "" +
                                        aggDataFrame["SEX"].astype(str) + "" +
                                        aggDataFrame["AGE_CAT"].astype(str)).str.upper()




# Dikkat! List comprehension ile customers_level_based değerleri oluşturulduktan sonra bu değerlerin tekilleştirilmesi gerekmektedir.
# Örneğin birden fazla şu ifadeden olabilir: USA_ANDROID_MALE_0_18. Bunları groupby'a alıp price ortalamalarını almak gerekmektedir.

agg_df = agg_df.groupby("customers_level_based").agg({"price": "mean"}).reset_index()
agg_df



####################################################################
# Görev 7: Yeni müşterileri (personaları) segmentlere ayırınız.
####################################################################

# - Yeni müşterileri (Örnek: USA_ANDROID_MALE_0_18) PRICE’a göre 4 segmente ayırınız.
# - Segmentleri SEGMENT isimlendirmesi ile değişken olarak agg_df’e ekleyiniz.
# - Segmentleri betimleyiniz (Segmentlere göre group by yapıp price mean, max, sum’larını alınız).

agg_df["segment"] = pd.qcut(agg_df["price"], 4, labels=["D","C","B","A"])
agg_df.groupby("segment").agg({"price": ["mean","max","sum"]})

####################################################################
# Görev 8: Yeni gelen müşterileri sınıflandırıp, ne kadar gelir getirebileceklerini tahmin ediniz.
####################################################################

# 33 yaşında ANDROID kullanan bir Türk kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?
# 35 yaşında IOS kullanan bir Fransız kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?

new_user = "TUR_ANDROID_FEMALE_ADULT"
agg_df[agg_df["customers_level_based"] == new_user]

new_user2 = "FRA_IOS_MALE_YOUNG"
agg_df[agg_df["customers_level_based"] == new_user2]



def predict_income(dataframe, country, source, sex, age):
    if age < 18:
        temp_age = "KID"
    elif 18 <= age and age < 23:
        temp_age = "YOUNG"
    elif 23 <= age and age < 30:
        temp_age = "ADULT"
    elif 30 <= age and age < 40:
        temp_age = "MATURE"
    else:
        temp_age = "OLD"

    user = country.upper() + "_" + source.upper() + "_" + sex.upper() + "_" + temp_age.upper()
    return dataframe[dataframe["customers_level_based"] == user]

predict_income(agg_df, "TUR", "IOS", "MALE", 25)