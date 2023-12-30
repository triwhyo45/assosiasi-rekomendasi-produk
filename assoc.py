import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import association_rules, apriori

df = pd.read_csv("restaurant-1-orders.csv")

df['Order Date'] = pd.to_datetime(df['Order Date'], format="%d/%m/%Y %H:%M")
df["month"] = df["Order Date"].dt.strftime('%B')
df["day"] = df["Order Date"].dt.day_name()
itemList = df['Item Name'].unique().tolist()

def getData(month="", day=""):
    data = df.copy()
    if month and day:
        filtered = data[(data["month"] == month) & (data["day"] == day)]
        return filtered if not filtered.empty else "No Result"
    elif month:
        filtered = data[data["month"] == month]
        return filtered if not filtered.empty else "No Result"
    elif day:
        filtered = data[data["day"] == day]
        return filtered if not filtered.empty else "No Result"
    else:
        return "No Result"

st.title("Rekomendasi produk yang relevan berdasarkan pesanan sebelumnya")

def user_input_feature():
    item = st.selectbox("Item Name", itemList)
    month = st.select_slider("Month", ['January','February','March','April','May','June','July','August','September','October','November','December'])
    day = st.select_slider("Day", ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])
    support = st.select_slider("Support", [0.05,0.06,0.07,0.08,0.09,0.1])
    return item,month,day,support

item, month, day, support = user_input_feature()
data = getData(month,day)

# Fungsi untuk mengonversi nilai dalam tabel pivot menjadi biner (0/1)
def encode(x):
    return 1 if x >= 1 else 0
  
if type(data) != type ("No Result") :
    itemCount = data.groupby(["Order Number", "Item Name"])["Item Name"].count().reset_index(name="Count")
    itemCountPivot = itemCount.pivot_table(index="Order Number", columns="Item Name", values="Count", aggfunc="sum").fillna(0)
    itemCountPivot = itemCountPivot.applymap(encode)
    frequentItems = apriori(itemCountPivot, min_support=support, use_colnames=True)
    metric = "lift"
    min_threshold = 1
    rules = association_rules(frequentItems, metric=metric, min_threshold=min_threshold)[["antecedents", "consequents", "support", "confidence", "lift"]]
    rules.sort_values("confidence", ascending=False, inplace=True)

def parse_list(x):
    x = list(x)
    if len(x) == 1:
        return x[0] 
    elif len(x) > 1:
        return ", ".join(x)

def return_item_df(item_antecedents):
    data_filtered = rules[(rules['antecedents'].apply(lambda x: item_antecedents in x))]
    if not data_filtered.empty:
        antecedent = parse_list(data_filtered['antecedents'].values[0])
        consequent = parse_list(data_filtered['consequents'].values[0])
        return [antecedent, consequent]
    else:
        return 0

if type(data) != type("No Result!"):
    if(return_item_df(item) != 0):
        st.markdown("Hasil Assosiasi : ")
        st.success(f"Jika Konsumen Membeli **{item}**, maka akan membeli **{return_item_df(item)[1]}** secara bersamaan")
    else:
        st.markdown("Hasil Assosiasi : ")
        st.warning(f"Nilai support terlalu tinggi atau konsumen hanya akan membeli **{item}**")

st.markdown("Support (nilai penunjang) adalah persentase kombinasi item tersebut dengan item lain dalam dataset")