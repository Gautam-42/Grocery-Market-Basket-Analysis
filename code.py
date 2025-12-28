import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


GROCERIES_PATH = "/Users/gautamkumarsingh/Desktop/coding/grocery market basket analysis/Groceries data.csv"
BASKET_PATH = "/Users/gautamkumarsingh/Desktop/coding/grocery market basket analysis/basket.csv"


df = pd.read_csv(GROCERIES_PATH)

print("Data loaded successfully")


df['itemDescription'] = (
    df['itemDescription']
    .astype(str)
    .str.lower()
    .str.strip()
)

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df.dropna(subset=['Member_number', 'itemDescription', 'Date'], inplace=True)

df['TransactionID'] = (
    df['Member_number'].astype(str) + "_" + df['Date'].astype(str)
)

print("\nDataset Summary:")
print("Total records:", len(df))
print("Unique customers:", df['Member_number'].nunique())
print("Unique items:", df['itemDescription'].nunique())
print("Unique transactions:", df['TransactionID'].nunique())


top_items = df['itemDescription'].value_counts().head(15)

plt.figure()
top_items.plot(kind='bar')
plt.title("Top 15 Most Sold Items")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

items_per_transaction = df.groupby('TransactionID')['itemDescription'].count()

plt.figure()
sns.histplot(items_per_transaction, bins=20)
plt.title("Items per Transaction Distribution")
plt.xlabel("Number of Items")
plt.tight_layout()
plt.show()

basket = (
    df.groupby(['TransactionID', 'itemDescription'])['itemDescription']
    .count()
    .unstack()
    .fillna(0)
)

basket = basket.applymap(lambda x: 1 if x > 0 else 0)

frequent_itemsets = apriori(
    basket,
    min_support=0.01,
    use_colnames=True
)

print("\nTop Frequent Itemsets:")
print(frequent_itemsets.sort_values(by="support", ascending=False).head())


rules = association_rules(
    frequent_itemsets,
    metric="lift",
    min_threshold=1.0
)

rules = rules.sort_values(by="confidence", ascending=False)

print("\nTop Association Rules:")
print(
    rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    .head(10)
)


snapshot_date = df['Date'].max() + pd.Timedelta(days=1)

rfm = df.groupby('Member_number').agg({
    'Date': lambda x: (snapshot_date - x.max()).days,
    'TransactionID': 'nunique',
    'itemDescription': 'count'
})

rfm.columns = ['Recency', 'Frequency', 'Monetary']

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

kmeans = KMeans(n_clusters=3, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

print("\nCustomer Cluster Summary:")
print(rfm.groupby('Cluster').mean())


pca = PCA(n_components=2)
rfm_pca = pca.fit_transform(rfm_scaled)

plt.figure()
plt.scatter(
    rfm_pca[:, 0],
    rfm_pca[:, 1],
    c=rfm['Cluster']
)
plt.title("Customer Segmentation (PCA View)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.tight_layout()
plt.show()

print("\nBusiness Insights:")
print("• High-support items must always be in stock.")
print("• High-lift item pairs should be bundled or placed together.")
print("• Loyal customers should receive rewards.")
print
