# batch_generate.py
import joblib
import pandas as pd

model_bundle = joblib.load("hybrid_gbdt_model.pkl")
model, scaler, features = model_bundle["model"], model_bundle["scaler"], model_bundle["features"]

# Giả sử có danh sách user trong hệ thống
user_list = pd.read_csv("all_users.csv")["CustomerId"].tolist()

results = []
for uid in user_list:
    ranked, _ = score_user(uid)   # hàm đã viết ở step 4.5
    if ranked is None: continue
    topn = ranked.head(10)[["CatalogEntryId","score"]]
    for _, row in topn.iterrows():
        results.append({"CustomerId": uid, "CatalogEntryId": row["CatalogEntryId"], "score": row["score"]})

df_out = pd.DataFrame(results)
df_out.to_csv("daily_recommendations.csv", index=False)
