import pandas as pd

# Load cleaned data
cleaned_data = pd.read_csv('data/Historical-transaction-cleaned-data.csv')
cleaned_data["invoice_id"] = cleaned_data["invoice_id"].astype(int)
cleaned_data["transaction_date"] = pd.to_datetime(cleaned_data["transaction_date"])

# Calculate sales
cleaned_data['sales'] = cleaned_data['item_price'] * cleaned_data['quantity_sold']

# Calculate the features per shop
features = cleaned_data.groupby('shop_id').agg(
    total_sales=('sales', 'sum'),
    total_quantity_sold=('quantity_sold', 'sum'),
    total_bills=('invoice_id', pd.Series.nunique),
    total_transactions=('transaction_date', 'count'),
    total_unique_items=('item_description', pd.Series.nunique),
    total_unique_customers=('customer_id', pd.Series.nunique),
    total_weeks=('transaction_date', lambda x: x.dt.isocalendar().week.nunique()),
    total_unique_dates=('transaction_date', lambda x: x.dt.weekday.nunique()),
)

# Load store info data
store_info = pd.read_csv('data/Store-info.csv')
store_info = store_info[['shop_id', 'shop_area_sq_ft']]

# Merge store area data with the features
combined_features = features.reset_index().merge(store_info, on='shop_id', how='left')

# Save the combined features as a new CSV file
combined_features.to_csv('data/Shop_Features.csv', index=False)
