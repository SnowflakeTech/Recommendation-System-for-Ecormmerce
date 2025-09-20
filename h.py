# Fix string normalization: use .str.strip() not .strip()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import textwrap

DATA_PATH = Path("/mnt/data/data.csv")
OUT_DIR = Path("/mnt/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_clean_transactions(path: Path) -> pd.DataFrame:
    df = pd.read_fwf(path, encoding="utf-8")
    df[["OrderGroupId", "CustomerId"]] = df["OrderGroupId CustomerId"].str.split(expand=True, n=1)
    df[["MarketId", "LineItemId"]] = df["MarketId LineItemId"].str.split(expand=True, n=1)
    df[["Epi_TaxCategoryId", "OrderCount"]] = df["Epi_TaxCategoryId OrderCount"].str.split(expand=True, n=1)
    df = df.drop(columns=["OrderGroupId CustomerId", "MarketId LineItemId", "Epi_TaxCategoryId OrderCount"])
    df["OrderGroupId"] = df["OrderGroupId"].astype(str)
    df = df[~df["OrderGroupId"].str.contains("-")].copy()
    for c in ["OrderTotal","SubTotal","ShippingTotal","TaxTotal","Quantity","PlacedPrice","ListPrice","LineItemDiscountAmount","OrderLevelDiscountAmount","ExtendedPrice","OrderCount","TotalQuantity","AvgPricePerItem"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["LineItemOrdering"] = pd.to_datetime(df["LineItemOrdering"], errors="coerce")
    for c in ["CustomerId","CatalogEntryId","ProductName","OrderStatus","MarketId","WarehouseCode"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df

tx = load_clean_transactions(DATA_PATH)
tx.to_csv(OUT_DIR / "clean_transactions.csv", index=False)

summary = {
    "n_rows": int(tx.shape[0]),
    "n_cols": int(tx.shape[1]),
    "date_min": tx["LineItemOrdering"].min(),
    "date_max": tx["LineItemOrdering"].max(),
    "n_customers": tx["CustomerId"].nunique(),
    "n_items": tx["CatalogEntryId"].nunique(),
    "n_orders": tx["OrderGroupId"].nunique(),
}

missing_pct = (tx.isna().mean() * 100).sort_values(ascending=False).reset_index()
missing_pct.columns = ["column","missing_%"]

numeric_cols_present = tx.select_dtypes(include=[np.number]).columns.tolist()
describe_num = tx[numeric_cols_present].describe().T.reset_index().rename(columns={"index":"metric"})

import caas_jupyter_tools
caas_jupyter_tools.display_dataframe_to_user("EDA: Dataset Summary", pd.DataFrame([summary]))
caas_jupyter_tools.display_dataframe_to_user("EDA: Missingness (%)", missing_pct)
caas_jupyter_tools.display_dataframe_to_user("EDA: Numeric Describe", describe_num)

valid_status = {"Completed","Shipped","Delivered","Paid"}
tx_valid = tx[tx["OrderStatus"].isin(valid_status)].copy()
tx_valid["month"] = tx_valid["LineItemOrdering"].dt.to_period("M")

monthly_tx = tx_valid.groupby("month").size().reset_index(name="transactions")
monthly_rev = tx_valid.groupby("month")["ExtendedPrice"].sum().reset_index(name="revenue")

plt.figure()
plt.plot(monthly_tx["month"].astype(str), monthly_tx["transactions"])
plt.title("Monthly Transactions")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUT_DIR / "monthly_transactions.png")
plt.show()

plt.figure()
plt.plot(monthly_rev["month"].astype(str), monthly_rev["revenue"])
plt.title("Monthly Revenue")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUT_DIR / "monthly_revenue.png")
plt.show()

plt.figure()
tx_valid["PlacedPrice"].dropna().plot(kind="hist", bins=30)
plt.title("Distribution: PlacedPrice")
plt.tight_layout()
plt.savefig(OUT_DIR / "dist_placedprice.png")
plt.show()

plt.figure()
np.log1p(tx_valid["ExtendedPrice"].dropna()).plot(kind="hist", bins=30)
plt.title("Distribution: log1p(ExtendedPrice)")
plt.tight_layout()
plt.savefig(OUT_DIR / "dist_log_spend.png")
plt.show()

top_products_qty = (tx_valid.groupby(["CatalogEntryId","ProductName"])["Quantity"]
                    .sum().sort_values(ascending=False).head(20).reset_index())
top_products_rev = (tx_valid.groupby(["CatalogEntryId","ProductName"])["ExtendedPrice"]
                    .sum().sort_values(ascending=False).head(20).reset_index())
caas_jupyter_tools.display_dataframe_to_user("Top 20 Products by Quantity", top_products_qty)
caas_jupyter_tools.display_dataframe_to_user("Top 20 Products by Revenue", top_products_rev)

cust_spend = (tx_valid.groupby("CustomerId")["ExtendedPrice"].sum()
              .sort_values(ascending=False).reset_index(name="total_spend"))
cust_orders = (tx_valid.groupby("CustomerId")["OrderGroupId"].nunique()
               .sort_values(ascending=False).reset_index(name="order_count"))
caas_jupyter_tools.display_dataframe_to_user("Top 20 Customers by Spend", cust_spend.head(20))
caas_jupyter_tools.display_dataframe_to_user("Top 20 Customers by Orders", cust_orders.head(20))

basket = (tx_valid.groupby("OrderGroupId")
          .agg(n_lines=("LineItemId","nunique"),
               qty_total=("Quantity","sum"),
               order_total=("ExtendedPrice","sum"))
          .reset_index())
caas_jupyter_tools.display_dataframe_to_user("Basket Metrics per Order", basket.head(50))

plt.figure()
basket["n_lines"].plot(kind="hist", bins=30)
plt.title("Distribution: Items per Order (distinct line items)")
plt.tight_layout()
plt.savefig(OUT_DIR / "dist_items_per_order.png")
plt.show()

plt.figure()
np.log1p(basket["order_total"]).plot(kind="hist", bins=30)
plt.title("Distribution: log1p(Order Total Spend)")
plt.tight_layout()
plt.savefig(OUT_DIR / "dist_log_order_total.png")
plt.show()

today = tx_valid["LineItemOrdering"].max()
rfm = (tx_valid.groupby("CustomerId")
       .agg(last_purchase=("LineItemOrdering","max"),
            frequency=("OrderGroupId","nunique"),
            monetary=("ExtendedPrice","sum"))
       .reset_index())
rfm["recency_days"] = (today - rfm["last_purchase"]).dt.days
caas_jupyter_tools.display_dataframe_to_user("RFM Table", rfm.head(50))

plt.figure()
rfm["recency_days"].plot(kind="hist", bins=30)
plt.title("Distribution: Customer Recency (days)")
plt.tight_layout()
plt.savefig(OUT_DIR / "dist_recency.png")
plt.show()

plt.figure()
rfm["frequency"].plot(kind="hist", bins=30)
plt.title("Distribution: Customer Frequency (orders)")
plt.tight_layout()
plt.savefig(OUT_DIR / "dist_frequency.png")
plt.show()

plt.figure()
np.log1p(rfm["monetary"]).plot(kind="hist", bins=30)
plt.title("Distribution: Customer Monetary (log1p spend)")
plt.tight_layout()
plt.savefig(OUT_DIR / "dist_monetary.png")
plt.show()

(OUT_DIR / "top_products_by_quantity.csv").write_text(top_products_qty.to_csv(index=False))
(OUT_DIR / "top_products_by_revenue.csv").write_text(top_products_rev.to_csv(index=False))
(OUT_DIR / "top_customers_by_spend.csv").write_text(cust_spend.to_csv(index=False))
(OUT_DIR / "top_customers_by_orders.csv").write_text(cust_orders.to_csv(index=False))
(OUT_DIR / "basket_metrics.csv").write_text(basket.to_csv(index=False))
(OUT_DIR / "rfm_table.csv").write_text(rfm.to_csv(index=False))

md = f"""
# EDA Summary

- Rows: {summary['n_rows']}
- Cols: {summary['n_cols']}
- Date range: {summary['date_min']} → {summary['date_max']}
- Unique customers: {summary['n_customers']}
- Unique items: {summary['n_items']}
- Unique orders: {summary['n_orders']}

Artifacts saved in /mnt/data:
- Cleaned transactions: clean_transactions.csv
- Monthly charts: monthly_transactions.png, monthly_revenue.png
- Distributions: dist_placedprice.png, dist_log_spend.png, dist_items_per_order.png, dist_log_order_total.png, dist_recency.png, dist_frequency.png, dist_monetary.png
- Tables: top_products_by_quantity.csv, top_products_by_revenue.csv, top_customers_by_spend.csv, top_customers_by_orders.csv, basket_metrics.csv, rfm_table.csv
"""
(Path("/mnt/data/EDA_SUMMARY.md")).write_text(textwrap.dedent(md).strip(), encoding="utf-8")

summary
