#!/usr/bin/env python
# coding: utf-8

# In[128]:


import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import create_engine


# In[129]:


# Load the data
# Make sure the file path is correct and the file exists
import os

file_path = '/home/kinas2k4/Documents/Recommendation-System-for-Ecormmerce/data/data.csv'
if not os.path.exists(file_path):
	raise FileNotFoundError(f"File not found: {file_path}. Please check the path and ensure the file exists.")

df_raw = pd.read_fwf(file_path, encoding="utf-8")
df_raw.head()


# In[130]:


df = df_raw.copy()

# Split OrderGroupId và CustomerId
df[["OrderGroupId", "CustomerId"]] = df["OrderGroupId CustomerId"].str.split(expand=True, n=1)

# Split MarketId và LineItemId
df[["MarketId", "LineItemId"]] = df["MarketId LineItemId"].str.split(expand=True, n=1)

# Split Epi_TaxCategoryId và OrderCount
df[["Epi_TaxCategoryId", "OrderCount"]] = df["Epi_TaxCategoryId OrderCount"].str.split(expand=True, n=1)

# Xóa cột gốc bị dính
df.drop(columns=["OrderGroupId CustomerId", "MarketId LineItemId", "Epi_TaxCategoryId OrderCount"], inplace=True)

print(df.head())


# In[131]:


# Chuyển sang string để xử lý
df["OrderGroupId"] = df["OrderGroupId"].astype(str)

# Bỏ dòng chứa "----"
df = df[~df["OrderGroupId"].str.contains("-")].copy()

print(df.head())


# In[132]:


# Các cột số
numeric_cols = [
    "OrderTotal", "SubTotal", "ShippingTotal", "TaxTotal",
    "Quantity", "PlacedPrice", "ListPrice",
    "LineItemDiscountAmount", "OrderLevelDiscountAmount",
    "ExtendedPrice", "OrderCount", "TotalQuantity", "AvgPricePerItem"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Cột thời gian
df["LineItemOrdering"] = pd.to_datetime(df["LineItemOrdering"], errors="coerce")

print(df.dtypes)


# In[133]:


df_rec = df[[
    "CustomerId", "CatalogEntryId", "ProductName", "Quantity", "PlacedPrice", "ListPrice",
    "ExtendedPrice", "OrderStatus", "LineItemOrdering"
]].copy()

print(df_rec.head(5))


# In[134]:


df_rec.isnull().sum()


# In[135]:


missing_pct = (df_rec.isna().mean() * 100).sort_values(ascending=False).reset_index()
missing_pct.columns = ["column","missing_%"]
missing_pct


# In[136]:


df_rec["CustomerId"] = df_rec["CustomerId"].fillna(0)
df_rec["CatalogEntryId"] = df_rec["CatalogEntryId"].fillna("Unknown")
df_rec["ProductName"] = df_rec["ProductName"].fillna("Unknown")
df_rec["Quantity"] = df_rec["Quantity"].fillna(0)
df_rec["ExtendedPrice"] = df_rec["ExtendedPrice"].fillna(0.0)
df_rec["PlacedPrice"] = df_rec["PlacedPrice"].fillna(0.0)
df_rec["ListPrice"] = df_rec["ListPrice"].fillna(0.0)
df_rec["OrderStatus"] = df_rec["OrderStatus"].fillna("Unknown")
df_rec["LineItemOrdering"] = df_rec["LineItemOrdering"].fillna(pd.Timestamp("1970-01-01"))


# In[137]:


df_rec.isnull().sum()


# In[138]:


# Ánh xạ CustomerId và CatalogEntryId sang chỉ số số nguyên liên tục
from sklearn.preprocessing import LabelEncoder
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()
df_rec['user_idx'] = user_encoder.fit_transform(df_rec['CustomerId'].astype(str))
df_rec['item_idx'] = item_encoder.fit_transform(df_rec['CatalogEntryId'].astype(str))

# Đảm bảo LineItemOrdering là kiểu datetime
df_rec['LineItemOrdering'] = pd.to_datetime(df_rec['LineItemOrdering'], errors='coerce')

# Thêm thuộc tính Day, Month, Year từ LineItemOrdering
df_rec['Day'] = df_rec['LineItemOrdering'].dt.day
df_rec['Month'] = df_rec['LineItemOrdering'].dt.month
df_rec['Year'] = df_rec['LineItemOrdering'].dt.year

df_rec[['CustomerId', 'user_idx', 'CatalogEntryId', 'item_idx', 'LineItemOrdering', 'Day', 'Month', 'Year']].head()


# ## EDA (Exploratory Data Analysis)

# In[139]:


summary = {
    "n_rows": int(df_rec.shape[0]),
    "n_customers": int(df_rec['CustomerId'].nunique()),
    "n_items": int(df_rec['CatalogEntryId'].nunique()),
    "n_interactions": int(df_rec.shape[0]),
    "min_quantity": int(df_rec['Quantity'].min()),
    "max_quantity": int(df_rec['Quantity'].max()),
    "avg_quantity": float(df_rec['Quantity'].mean()),
    "min_price": float(df_rec['ExtendedPrice'].min()),
    "max_price": float(df_rec['ExtendedPrice'].max()),
    "avg_price": float(df_rec['ExtendedPrice'].mean()),
    "date_range": (df_rec['LineItemOrdering'].min(), df_rec['LineItemOrdering'].max())
}
summary


# In[140]:


# Numeric Describe
numeric_cols_present = df_rec.select_dtypes(include=[np.number]).columns.tolist()
describe_num = df_rec[numeric_cols_present].describe().T.reset_index().rename(columns={"index":"metric"})
describe_num


# In[141]:


valid_status = {"Completed","Shipped","Delivered","Paid"}
transaction_valid = df_rec[df_rec["OrderStatus"].isin(valid_status)].copy()
transaction_valid["Month"] 


# In[142]:


monthly_tx = transaction_valid.groupby("Month").size().reset_index(name="transactions") 
monthly_rev = transaction_valid.groupby("Month")["ExtendedPrice"].sum().reset_index(name="revenue")


# In[143]:


plt.figure()
plt.plot(monthly_tx["Month"].astype(str), monthly_tx["transactions"])
plt.title("Monthly Transactions")
plt.tight_layout()
plt.show()


# In[144]:


plt.figure()
plt.plot(monthly_rev["Month"].astype(str), monthly_rev["revenue"])
plt.title("Monthly Revenue")
plt.tight_layout()
plt.show()


# In[145]:


top_products_qty = (transaction_valid.groupby(["CatalogEntryId","ProductName"])["Quantity"].sum().sort_values(ascending=False).head(5).reset_index())
top_products_rev = (transaction_valid.groupby(["CatalogEntryId","ProductName"])["ExtendedPrice"].sum().sort_values(ascending=False).head(5).reset_index())
print("Top 5 Products by Quantity", top_products_qty)
print("Top 5 Products by Revenue", top_products_rev)


# In[146]:


# Tính bảng RFM (Recency, Frequency, Monetary)
import datetime

# Ngày tham chiếu là ngày lớn nhất trong dữ liệu + 1
snapshot_date = df_rec['LineItemOrdering'].max() + pd.Timedelta(days=1)

rfm = df_rec.groupby('CustomerId').agg({
    'LineItemOrdering': lambda x: (snapshot_date - x.max()).days,  # Recency
    'OrderStatus': 'count',  # Frequency (số lần mua)
    'ExtendedPrice': 'sum'   # Monetary (tổng chi tiêu)
}).reset_index()

rfm.columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary']
rfm = rfm.sort_values('Recency')
rfm.head(20)


# In[147]:


# Tạo ma trận CSR cho user-item (implicit feedback)
from scipy import sparse
from scipy.sparse import save_npz
import os

# Đảm bảo DATA_DIR là thư mục data trong workspace
DATA_DIR = '/home/kinas2k4/Documents/Recommendation-System-for-Ecormmerce/data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Chỉ lấy các giao dịch hợp lệ (ví dụ: đã lọc trạng thái)
tx_valid = transaction_valid.copy()
tx_valid["implicit_score"] = 1.0 + np.log1p(tx_valid["ExtendedPrice"].fillna(0.0)) + 0.2 * tx_valid["Quantity"].fillna(0.0)

agg = (tx_valid.groupby(["CustomerId", "CatalogEntryId"], as_index=False)
       .agg(score=("implicit_score", "sum"),
            qty=("Quantity", "sum"),
            spend=("ExtendedPrice", "sum")))

user_ids = agg["CustomerId"].unique()
item_ids = agg["CatalogEntryId"].unique()
uid2ix = {u: i for i, u in enumerate(user_ids)}
iid2ix = {it: i for i, it in enumerate(item_ids)}
agg["u_idx"] = agg["CustomerId"].map(uid2ix)
agg["i_idx"] = agg["CatalogEntryId"].map(iid2ix)

num_users = len(user_ids)
num_items = len(item_ids)
mat = sparse.csr_matrix((agg["score"].values, (agg["u_idx"].values, agg["i_idx"].values)),
                        shape=(num_users, num_items), dtype=np.float32)

# save_npz(os.path.join(DATA_DIR, "user_item_matrix.npz"), mat)
# pd.Series(user_ids, name="CustomerId").to_csv(os.path.join(DATA_DIR, "user_index_to_id.csv"), index=False)
# pd.Series(item_ids, name="CatalogEntryId").to_csv(os.path.join(DATA_DIR, "item_index_to_id.csv"), index=False)
# agg.to_csv(os.path.join(DATA_DIR, "user_item_interactions.csv"), index=False)

# Xem trước ma trận top 10x10
top_users = (agg.groupby("u_idx")["score"].sum().sort_values(ascending=False).head(10).index.tolist())
top_items = (agg.groupby("i_idx")["score"].sum().sort_values(ascending=False).head(10).index.tolist())
preview = (agg[agg["u_idx"].isin(top_users) & agg["i_idx"].isin(top_items)]
           .pivot_table(index="u_idx", columns="i_idx", values="score", fill_value=0.0)
           .reset_index())

preview.head(10)


# ## Heuristics & Baseline|

# In[148]:


popularity = (tx_valid.groupby("CatalogEntryId")["Quantity"].sum().sort_values(ascending=False).head(10))
print(popularity)


# In[149]:


cutoff = tx_valid["LineItemOrdering"].max() - pd.Timedelta(days=30)
trending = (
    tx_valid[tx_valid["LineItemOrdering"] >= cutoff]
    .groupby("CatalogEntryId")["Quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)
print(trending)


# # Collaborative Filtering

# ## Item–Item kNN (Cosine / Jaccard / BM25)

# In[150]:


from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


# In[151]:


mat = sparse.load_npz("/home/kinas2k4/Documents/Recommendation-System-for-Ecormmerce/data/user_item_matrix.npz")
item_ids = pd.read_csv("/home/kinas2k4/Documents/Recommendation-System-for-Ecormmerce/data/item_index_to_id.csv")["CatalogEntryId"].tolist()
num_items = len(item_ids)
num_items


# In[152]:


# Cosine similarity
item_norms = sparse.linalg.norm(mat, axis=0)
mat_normed = mat / item_norms
sim_cosine = (mat_normed.T @ mat_normed).toarray()
sim_cosine


# In[153]:


# Jaccard similarity
mat_bin = mat.copy()
mat_bin.data = np.ones_like(mat_bin.data)
intersection = (mat_bin.T @ mat_bin).toarray()  # |U_i ∩ U_j|
col_sums = np.array(mat_bin.sum(axis=0)).ravel()
union = col_sums[:, None] + col_sums[None, :] - intersection
sim_jaccard = intersection / np.maximum(union, 1e-9)
sim_jaccard


# In[154]:


# BM25 weighting + cosine
k1, b = 1.2, 0.75
N = mat.shape[0]
idf = np.log((N - np.array((mat > 0).sum(axis=0)).ravel() + 0.5) /
             (np.array((mat > 0).sum(axis=0)).ravel() + 0.5))
idf = np.maximum(idf, 0)

# document length (user profile length)
dl = np.array(mat.sum(axis=1)).ravel()
avgdl = dl.mean()

# Apply BM25 weighting
rows, cols = mat.nonzero()
data = []
for r, c, v in zip(rows, cols, mat.data):
    denom = v + k1 * (1 - b + b * dl[r] / avgdl)
    weight = idf[c] * (v * (k1 + 1)) / denom
    data.append(weight)
mat_bm25 = sparse.csr_matrix((data, (rows, cols)), shape=mat.shape)


# In[155]:


# Cosine similarity on BM25 weighted
item_norms_bm25 = sparse.linalg.norm(mat_bm25, axis=0)
mat_bm25_norm = mat_bm25 / item_norms_bm25
sim_bm25 = (mat_bm25_norm.T @ mat_bm25_norm).toarray()


# In[156]:


df_cosine = pd.DataFrame(sim_cosine, index=item_ids, columns=item_ids)
df_jaccard = pd.DataFrame(sim_jaccard, index=item_ids, columns=item_ids)
df_bm25 = pd.DataFrame(sim_bm25, index=item_ids, columns=item_ids)


# In[157]:


df_cosine.head()


# In[158]:


df_jaccard.head()


# In[159]:


df_bm25.head()


# In[160]:


example_item = item_ids[0]
sim_scores = df_cosine.loc[example_item].drop(example_item).sort_values(ascending=False).head(5)
print(f"Top-5 similar items to {example_item} (Cosine):")
print(sim_scores)


# In[161]:


example_item = item_ids[0]
sim_scores = df_jaccard.loc[example_item].drop(example_item).sort_values(ascending=False).head(5)
print(f"Top-5 similar items to {example_item} (Jaccard):")
print(sim_scores)


# In[162]:


example_item = item_ids[0]
sim_scores = df_bm25.loc[example_item].drop(example_item).sort_values(ascending=False).head(5)
print(f"Top-5 similar items to {example_item} (BM25):")
print(sim_scores)


# ## User–User kNN (Cosine / Jaccard / BM25)

# In[163]:


user_ids = pd.read_csv("/home/kinas2k4/Documents/Recommendation-System-for-Ecormmerce/data/user_index_to_id.csv")["CustomerId"].tolist()
num_users = len(user_ids)
num_users


# In[164]:


# Tính norm của mỗi user (hàng)
user_norms = sparse.linalg.norm(mat, axis=1)

# Chuyển thành cột (2D) để broadcasting hợp lệ
user_norms = np.array(user_norms).reshape(-1, 1)

# Chia từng hàng cho norm tương ứng
mat_normed = mat.multiply(1.0 / (user_norms + 1e-9))

# Sau đó tính cosine similarity
sim_cosine = (mat_normed @ mat_normed.T).toarray()


# In[165]:


mat_bin = mat.copy()
mat_bin.data = np.ones_like(mat_bin.data)

intersection = (mat_bin @ mat_bin.T).toarray()  # |I_u ∩ I_v|
row_sums = np.array(mat_bin.sum(axis=1)).ravel()
union = row_sums[:, None] + row_sums[None, :] - intersection
sim_jaccard = intersection / np.maximum(union, 1e-9)
sim_jaccard


# In[166]:


k1, b = 1.2, 0.75
N = mat.shape[1]  # number of items
df = np.array((mat > 0).sum(axis=0)).ravel()  # document frequency = how many users per item
idf = np.log((N - df + 0.5) / (df + 0.5))
idf = np.maximum(idf, 0)

dl = np.array(mat.sum(axis=1)).ravel()  # doc length = items per user
avgdl = dl.mean()

rows, cols = mat.nonzero()
data = []
for r, c, v in zip(rows, cols, mat.data):
    denom = v + k1 * (1 - b + b * dl[r] / avgdl)
    weight = idf[c] * (v * (k1 + 1)) / denom
    data.append(weight)
mat_bm25 = sparse.csr_matrix((data, (rows, cols)), shape=mat.shape)
user_norms_bm25 = sparse.linalg.norm(mat_bm25, axis=1)
user_norms_bm25 = np.array(user_norms_bm25).reshape(-1, 1)  # <- fix quan trọng

mat_bm25_norm = mat_bm25.multiply(1.0 / (user_norms_bm25 + 1e-9))
sim_bm25 = (mat_bm25_norm @ mat_bm25_norm.T).toarray()
sim_bm25


# In[167]:


df_cosine = pd.DataFrame(sim_cosine, index=user_ids, columns=user_ids)
df_jaccard = pd.DataFrame(sim_jaccard, index=user_ids, columns=user_ids)
df_bm25 = pd.DataFrame(sim_bm25, index=user_ids, columns=user_ids)


# In[168]:


df_cosine.head()


# In[169]:


df_jaccard.head()


# In[176]:


example_user = user_ids[0]
sim_scores = df_cosine.loc[example_user].drop(example_user).sort_values(ascending=False).head(5)
print(f"Top-5 similar users to {example_user} (Cosine):")
print(sim_scores)


# In[177]:


from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=50, random_state=42)
mat_svd = svd.fit_transform(mat)
mat_svd_norm = mat_svd / np.linalg.norm(mat_svd, axis=1, keepdims=True)
sim_svd = mat_svd_norm @ mat_svd_norm.T
sim_svd


# In[179]:


# Giả sử sim_svd là user-user similarity (mat = user_item_matrix)
user_ids = list(uid2ix.keys())  # mapping từ trước

example_user = user_ids[0]   # lấy user đầu tiên
u_idx = uid2ix[example_user]

# Lấy top-5 similar users
sim_scores = pd.Series(sim_svd[u_idx], index=user_ids).drop(example_user)
top5 = sim_scores.sort_values(ascending=False).head(5).reset_index()
top5.columns = ["CustomerId","similarity"]

print(f"Top-5 similar users to {example_user}:")
display(top5)


# In[184]:


# Sử dụng embedding của item từ SVD (svd.components_.T)
item_ids = list(iid2ix.keys())
item_emb = svd.components_.T  # shape: (num_items, n_components)

example_item = item_ids[0]
i_idx = iid2ix[example_item]

# Tính cosine similarity giữa example_item và các item khác trong latent space
from sklearn.metrics.pairwise import cosine_similarity

sims = cosine_similarity(item_emb[i_idx].reshape(1, -1), item_emb).flatten()
sim_scores = pd.Series(sims, index=item_ids).drop(example_item)
top5 = sim_scores.sort_values(ascending=False).head(5).reset_index()
top5.columns = ["CatalogEntryId","similarity"]

# Ghép thêm thông tin sản phẩm
top5 = top5.merge(items[["CatalogEntryId","ProductName","Price"]],
                  on="CatalogEntryId", how="left")

print(f"Top-5 similar items to {example_item} (SVD latent space):")
display(top5)


# ## Content-Based Filtering

# In[170]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack


# In[171]:


items = (tx_valid.groupby("CatalogEntryId")
           .agg(ProductName=("ProductName","first"),
                Price=("PlacedPrice","mean"))
           .reset_index())
items["ProductName"] = items["ProductName"].fillna("").astype(str)
items["Price"] = items["Price"].fillna(items["Price"].median())

print(f"Unique products: {items.shape[0]}")


# In[172]:


# TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
tfidf = vectorizer.fit_transform(items["ProductName"])


# In[173]:


# 3) Chuẩn hóa Price (0-1)
scaler = MinMaxScaler()
price_scaled = scaler.fit_transform(items[["Price"]])

# 4) Ghép TF-IDF + Price
X_content = hstack([tfidf, price_scaled])

# 5) Tính cosine similarity
sim_content = cosine_similarity(X_content)


# In[174]:


df_sim_content = pd.DataFrame(sim_content, index=items["CatalogEntryId"], columns=items["CatalogEntryId"])
df_sim_content.to_csv("/home/kinas2k4/Documents/Recommendation-System-for-Ecormmerce/data/item_content_similarity.csv", index=True)
df_sim_content.head(5)


# In[175]:


def get_similar_items_with_info(item_id, sim_matrix, items, topn=5):
    if item_id not in sim_matrix.index:
        return pd.DataFrame(columns=["CatalogEntryId","ProductName","Price","similarity"])
    scores = sim_matrix.loc[item_id].drop(item_id).sort_values(ascending=False).head(topn)
    df = scores.reset_index()
    df.columns = ["CatalogEntryId","similarity"]
    df = df.merge(items[["CatalogEntryId","ProductName","Price"]], on="CatalogEntryId", how="left")
    return df[["CatalogEntryId","ProductName","Price","similarity"]]

# Demo
top5_info = get_similar_items_with_info(example_item, df_sim_content, items, topn=5)
display(top5_info)

