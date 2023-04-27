import pandas as pd

transaction_data = pd.read_csv('data/Historical-transaction-data.csv')

# Drop rows with less than 1 values for quantity_sold
initial_quantity_sold_count = len(transaction_data)
transaction_data = transaction_data[transaction_data['quantity_sold'] > 0]
print("Removed rows with less than 1 quantity_sold value: ", initial_quantity_sold_count - len(transaction_data))

# Fill missing item_description values
intitial_unique_items = len(transaction_data['item_description'].unique()) - 1 # without null
def fill_item_description(row):
    if pd.isnull(row['item_description']):
        new_name = f"product_{row['shop_id']}__{row['item_price']}"
        return new_name
    else:
        return row['item_description']

transaction_data['item_description'] = transaction_data.apply(fill_item_description, axis=1)
print("No of item_description values generated: ", len(transaction_data['item_description'].unique()) - intitial_unique_items)

# Fill missing invoice_id values
max_invoice_id = transaction_data['invoice_id'].max()
intitial_unique_items = len(transaction_data['invoice_id'].unique()) - 1 # without null
new_invoice_id = max_invoice_id + 1
invoice_id_dict = {}

def fill_invoice_id(row):
    global new_invoice_id, invoice_id_dict
    if pd.isnull(row['invoice_id']):
        key = (row['transaction_date'], row['customer_id'], row['shop_id'])
        if key not in invoice_id_dict:
            invoice_id_dict[key] = new_invoice_id
            new_invoice_id += 1
        return invoice_id_dict[key]
    else:
        return row['invoice_id']

transaction_data['invoice_id'] = transaction_data.apply(fill_invoice_id, axis=1)
print("No of invoice_ids generated: ", len(transaction_data['invoice_id'].unique()) - intitial_unique_items)

# Save cleaned data to a new CSV file
transaction_data.to_csv('data/Historical-transaction-cleaned-data.csv', index=False)
