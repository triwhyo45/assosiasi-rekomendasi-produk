### Nama : Tri Wahyono

### Nim : 211351145

### Kelas : Malam A

## Domain Proyek

Dalam dataset pesanan dari restoran India di London, ada beberapa pendekatan yang dapat digunakan untuk membuat model asosiasi untuk rekomendasi produk berdasarkan variabel yang ada. Model asosiasi bertujuan untuk menemukan hubungan atau pola yang ada di antara produk-produk yang sering dibeli bersamaan oleh pelanggan.

## Business Understanding

Aplikasi ini dikembangkan untuk meningkatkan pengalaman pelanggan dan efisiensi operasional. Dengan memahami pola pembelian pelanggan, aplikasi akan memberikan rekomendasi produk yang relevan berdasarkan pesanan sebelumnya. Ini akan membantu restoran meningkatkan penjualan melalui strategi penawaran yang lebih personal kepada pelanggan.

### Problem Statement

-   Restoran India di London belum sepenuhnya memanfaatkan potensi penjualan dari pola pembelian pelanggan.
-   Pelanggan mungkin mengalami kesulitan dalam membuat keputusan saat memesan, memperlambat proses pemesanan dan mengurangi potensi penjualan tambahan.

### Goals

-   Meningkatkan pendapatan restoran dengan mengoptimalkan pola pembelian pelanggan.
-   Mempercepat proses pemesanan dengan memberikan rekomendasi produk yang sesuai kepada pelanggan.

### Solution Statements

-   Membangun aplikasi yang memungkinkan pelanggan untuk melihat rekomendasi produk yang relevan saat mereka melakukan pemesanan, mempercepat proses dan meningkatkan peluang cross-selling.

## Data Understanding

Dataset ini merupakan kumpulan data dari dua restoran India di London, UK, yang tersedia dalam format CSV. Informasi tersebut berisi detail tentang pesanan yang dibuat di restoran tersebut. Setiap baris dalam file CSV mewakili satu produk yang dipesan dalam suatu pesanan. Jumlah total baris data saat ini adalah sekitar 200 ribu, yang mengindikasikan bahwa terdapat sekitar 33 ribu pesanan dari kedua restoran.

Dataset : [Takeaway Food Orders](https://www.kaggle.com/datasets/henslersoftware/19560-indian-takeaway-orders)

### Variabel-variabel pada datasets Takeaway Food Orders adalah sebagai berikut:

| Nama Variabel  | Description                                                | Tipe           |
| -------------- | ---------------------------------------------------------- | -------------- |
| Order Number:  | Nomor identifikasi unik untuk setiap pesanan               | int64          |
| Order Date     | Tanggal saat pesanan dibuat.                               | datetime64[ns] |
| Item Name      | Nama Item                                                  | object         |
| Quantity       | Jumlah produk yang dipesan dalam satu transaksi.           | int64          |
| Product Price  | Harga per unit dari produk dalam pesanan.                  | int64          |
| Total Products | Total keseluruhan produk yang terdapat dalam satu pesanan. | int64          |

## Data Preparation

Pada proses persiapan, hal yang pertama kita lakukan adalah import dataset yang akan kita gunakan

### Import Dataset

```py
from google.colab import files
files.upload()
```

```py
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle
```

```py
!kaggle datasets download -d henslersoftware/19560-indian-takeaway-orders
```

```py
!mkdir 19560-indian-takeaway-orders
!unzip 19560-indian-takeaway-orders.zip -d datasets
!ls datasets
```

### Import library

Lalu kita akan import library yang akan digunakan

```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from mlxtend.frequent_patterns import association_rules, apriori

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
```

### Data discovery

Pertama kita buat dahulu dataframenya dari datasets yang telah kita import / download diatas

```py
df = pd.read_csv('/content/datasets/restaurant-1-orders.csv', encoding="unicode_escape")
```

Tampilkan 5 baris pertama pada dataframe

```py
df.head()
```

Tampilkan informasi pada dataframe seperti tipe data, jumlah kolom, dan nama kolom

```py
df.info()
```

Lihat shape atau bentuk dataframe, disini kita memiliki sekitar 74818 baris dan 6 kolom

```py
df.shape
```

Mari kita lihat jumlah data unik pada setiap kolom, disni kita lihat ada sekitar 13397 order number atau transaksi yang telah terjadi

```py
df.nunique()
```

### EDA

Peratama kita lihat 10 produk terlaris yang berupa barplot

```py
plt.figure(figsize=(16,8))

top_products = df['Item Name'].value_counts().head(10)
sns.barplot(x=top_products.index, y=top_products.values, palette='viridis')
plt.xlabel("")
plt.ylabel("")
plt.xticks(size = 12, rotation = 90)
plt.title("10 Produk yang paling laris", size = 18)
plt.show
```

![10produklaris](https://github.com/triwhyo45/assosiasi-rekomendasi-produk/assets/155219695/2de15cb4-29d1-4533-a8eb-dbb6b35c06fb)

Pilau rice menjadi makanan nomor 1 yang paling laris terjual dengan jumlah pembelian lebih dari 4000 kali

Selanjutnya menampilkan tren jumlah pesanan per bulan selama beberapa tahun

```py
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Month'] = df['Order Date'].dt.month
df['Year'] = df['Order Date'].dt.year

monthly_orders = df.groupby(['Year', 'Month']).size().reset_index(name='Orders')
plt.figure(figsize=(10, 6))
sns.lineplot(data=monthly_orders, x='Month', y='Orders', hue='Year', marker='o', palette='Set2')
plt.title('Monthly Orders Trend')
plt.xlabel('Month')
plt.ylabel('Number of Orders')
plt.xticks(range(1, 13))
plt.legend(title='Year')
plt.show()
```

![ordertrend](https://github.com/triwhyo45/assosiasi-rekomendasi-produk/assets/155219695/eb844bfe-13e8-460d-a6a1-3c67f7dc56ce)

Dapat kita lihat bahwa jumlah pesanan memiliki pola yang fluktuatif. Jumlah pesanan cenderung meningkat pada bulan-bulan menjelang tahun baru, dan menurun pada bulan-bulan setelah tahun baru.

Menampilkan jumlah order perhari

```py
df['name day'] = df['Order Date'].dt.day_name()

plt.figure(figsize=(8, 6))
sns.countplot(x='name day', data=df, palette='rocket')
plt.title('Number of Orders per day')
plt.xlabel('day')
plt.ylabel('Number of Orders')
plt.show()
```

![dailyorder](https://github.com/triwhyo45/assosiasi-rekomendasi-produk/assets/155219695/95022830-773a-40b7-ba05-e3e466f5f0ce)

Bisa kita lihat puncaknya jumlah order perhari di dominasi oleh hari sabtu dan jumlah

Menampilkan jumlah order perbulan

```py
plt.figure(figsize=(8, 6))
sns.countplot(x='Month', data=df, palette='rocket')
plt.title('Number of Orders per Month')
plt.xlabel('Month')
plt.ylabel('Number of Orders')
plt.show()
```

![monthlyorder](https://github.com/triwhyo45/assosiasi-rekomendasi-produk/assets/155219695/4a20ff9f-3cec-411d-aa0e-2cd43a767da3)

Jumlah order perbulan di dominasi oleh bulan mei, julli dan desember

Menampilkan jumlah order pertahun

```py
plt.figure(figsize=(8, 6))
sns.countplot(x='Year', data=df, palette='rocket')
plt.title('Number of Orders per Year')
plt.xlabel('Year')
plt.ylabel('Number of Orders')
plt.show()
```

![yearlyorder](https://github.com/triwhyo45/assosiasi-rekomendasi-produk/assets/155219695/c4c9abad-4048-49cb-9df6-e5ea4a052474)

Bisa kita lihat jumlah order terbanyak terjadi pada tahun 2018 tetapi menurun pada tahun 2019 bahkan dibawah tahun 2017 ini artinya ada hal yang harus di evaluasi oleh pihak restoran

Menampilkan boxplot distribusi harga, dapat disimpulkan bahwa sebagian besar produk memiliki harga yang relatif murah.

```py
plt.figure(figsize=(8, 6))
sns.boxplot(y='Product Price', data=df, palette='pastel')
plt.title('Distribution of Product Prices')
plt.ylabel('Product Price')
plt.show()
```

![boxplot](https://github.com/triwhyo45/assosiasi-rekomendasi-produk/assets/155219695/fabd3562-208f-4a5a-aa5a-980f74abd25f)

### Preprocessing

Menambahkan kolom "month", "day", dan "hour", dengan nilai-nilai yang diambil dari kolom "Order Date", lalu menampilkan 5 isi dataframe

```py
df["month"] = df["Order Date"].dt.month
df["day"] = df["Order Date"].dt.weekday
df["hour"] = df["Order Date"].dt.hour
df.head()
```

Membuat data frame baru yang berisi Order Number, Item name dan quantity

```py
data = df[["Order Number", "Item Name", "Quantity"]].copy()
```

Membuat sebuah pivot table dari dataframe data yang menghitung jumlah Quantity untuk setiap Item name dalam setiap order number

```py
item_count_pivot = data.pivot_table(index="Order Number", columns="Item Name", values="Quantity", aggfunc="sum").fillna(0)
```

Konversi tipe data dari pivot table item_count_pivot menjadi int32

```py
item_count_pivot = item_count_pivot.astype("int32")
item_count_pivot.head()
```

Encode data yang kurang dari sama dengan nol menjadi nol dan lebih dari sama dengan 1 menjadi 1

```py
def encode(x):
  if x <= 0:
    return 0
  elif x >= 1:
    return 1

item_count_pivot = item_count_pivot.applymap(encode)
item_count_pivot.head()
```

## Modeling

Menggunakan Algoritma Apriori untuk menemukan kumpulan item yang sering muncul berdasarkan nilai support

```py
support = 0.01
frequent_items = apriori(item_count_pivot, min_support=support, use_colnames=True)
frequent_items.sort_values("support", ascending=False).head(10)
```

## Visualisasi hasil algoritma

Akhirnya, jadi hasil dari algoritma apriori kita mendapatkan baris-baris data yang memilki aturan aturan sepert support, confidence dll

```py
metric = "lift"
min_threshold = 1

rules = association_rules(frequent_items, metric=metric, min_threshold=min_threshold)[["antecedents", "consequents", "support", "confidence", "lift"]]

rules.sort_values("confidence", ascending=False, inplace=True)
rules.head(15)
```

![rules](https://github.com/triwhyo45/assosiasi-rekomendasi-produk/assets/155219695/f86c370c-0895-4181-a67c-5b8b07541215)

## Evaluation

Secara umum, hasil algoritma apriori menunjukkan bahwa saus onion chutney, mint sauce, dan mango chutney sering dibeli bersama dengan plain papadum.

## Deployment

[Asosiasi Rekomendasi Produk](https://asosiasi-rekomendasi.streamlit.app/)


![streamlit](https://github.com/triwhyo45/assosiasi-rekomendasi-produk/assets/155219695/3aff4192-e303-4a8f-a969-b2851bb982f0)

