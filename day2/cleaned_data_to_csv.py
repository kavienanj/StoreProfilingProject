import pandas as pd

transaction_data = pd.read_csv('data/Historical-transaction-data.csv')
print("Initial rows: ", len(transaction_data))

# Drop all rows with null values (item_description, invoice_id)
initial_lines = len(transaction_data)
transaction_data = transaction_data.dropna()
print("Removed lines with null values:", initial_lines - len(transaction_data))

# Drop rows with oultier values for item_price and > 0
initial_lines = len(transaction_data)
low_limit = 0
upper_limit = transaction_data["item_price"].quantile(0.98) # 700
transaction_data = transaction_data[(transaction_data["item_price"] > low_limit) & (transaction_data["item_price"] < upper_limit)]
print("Removed rows with outlier item_price value: ", initial_lines - len(transaction_data))

# Drop rows with oultier values for quantity_sold and > 0
initial_lines = len(transaction_data)
low_limit = 0
upper_limit = transaction_data["quantity_sold"].quantile(0.999) # 10
transaction_data = transaction_data[(transaction_data["quantity_sold"] > low_limit) & (transaction_data["quantity_sold"] < upper_limit)]
print("Removed rows with outlier quantity_sold value: ", initial_lines - len(transaction_data))

print("Writing rows: ", len(transaction_data))
# Save cleaned data to a new CSV file
transaction_data.to_csv('data/Historical-transaction-cleaned-data.csv', index=False)
