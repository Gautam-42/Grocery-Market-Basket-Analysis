# Grocery Market Basket Analysis & Customer Segmentation

## Objective
This project analyzes grocery transaction data to identify frequently purchased items using Market Basket Analysis and segments customers using clustering techniques. The goal is to help store owners improve inventory management and promotional strategies.

## Dataset
- Groceries data.csv
- basket.csv

## Techniques Used
- Exploratory Data Analysis (EDA)
- Apriori Algorithm
- Association Rule Mining
- RFM Model
- K-Means Clustering
- PCA for visualization

## Results
### Top Selling Items
![Top Items](results/top_items.png)

### Customer Segmentation
![Customer Clusters](results/customer_clusters.png)

## Business Insights
- High-support items should always be stocked.
- Frequently co-purchased items can be bundled.
- Loyal customers should be rewarded.
- At-risk customers should be targeted with offers.

## How to Run
```bash
pip install -r requirements.txt
python code.py
