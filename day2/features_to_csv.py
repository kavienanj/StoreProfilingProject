import pandas as pd

# Load cleaned data
cleaned_data = pd.read_csv('./data/Historical-transaction-cleaned-data.csv')
cleaned_data["invoice_id"] = cleaned_data["invoice_id"].astype(int)
cleaned_data["transaction_date"] = pd.to_datetime(cleaned_data["transaction_date"])

# Calculate sales
cleaned_data['sales'] = cleaned_data['item_price'] * cleaned_data['quantity_sold']
cleaned_data['weekend_transaction'] = cleaned_data['transaction_date'].dt.weekday > 4

# Calculate the features per shop
features = cleaned_data.groupby('shop_id').agg(
    total_sales=('sales', 'sum'),
    total_quantity_sold=('quantity_sold', 'sum'),
    mean_item_price=('item_price', 'mean'),
    total_bills=('invoice_id', pd.Series.nunique),
    total_transactions=('transaction_date', 'count'),
    total_items=('item_description', pd.Series.nunique),
    total_customers=('customer_id', pd.Series.nunique),
    weekday_transactions=('weekend_transaction', lambda x: x.value_counts()[False]),
    weekend_transactions=('weekend_transaction', lambda x: x.value_counts()[True]),
)

# Additional features, refer feature_selection.ipynb
features['product_customer_per_transaction'] = features['total_customers'] * features['total_transactions']
features['bills_per_customer'] = features['total_bills'] / features['total_customers']

# Load store info data
store_info = pd.read_csv('./data/Store-info.csv')
store_info = store_info[['shop_id', 'shop_area_sq_ft']]

# Merge store area data with the features
combined_features = features.reset_index().merge(store_info, on='shop_id', how='left')

# Save the combined features as a new CSV file
combined_features.to_csv('./data/Shop_Features.csv', index=False)
