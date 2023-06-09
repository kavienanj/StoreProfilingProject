import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the shop_features data
shop_features = pd.read_csv('./data/Shop_Features.csv')

# Load the store_info data
store_info = pd.read_csv('./data/Store-info.csv')
store_info = store_info[['shop_id', 'shop_profile']]
store_info = store_info.dropna(subset=['shop_profile'])
shop_profile_mapping = {'Low': 1, 'Moderate': 2, 'High': 3}
store_info['shop_profile'] = store_info['shop_profile'].replace(shop_profile_mapping)

# Merge the shop_features and store_info DataFrames
data = shop_features.merge(store_info, on='shop_id', how='left').dropna()

# Select the desired features and the target variable
selected_features = ['total_quantity_sold', 'total_unique_customers', 'total_unique_items']
X = data[selected_features]
y = data['shop_profile']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=131)

# Train the RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=111, random_state=190)
classifier.fit(X_train, y_train)

# Predict the shop profiles for the test data
y_pred = classifier.predict(X_test)

# Evaluate the model
print("\n * train_test_split predictions * \n")
score = accuracy_score(y_test, y_pred)
print("Accuracy:", score)
print("Classification Report:\n", classification_report(y_test, y_pred))

# ---------------------------------------------------------------
# Load the test data
test_data = pd.read_csv('./data/Testing-data.csv')

# Merge the test data with the shop_features DataFrame to get the shop_profile column
test_data = test_data.merge(shop_features, on='shop_id', how='left')
print(f"Predicting shop profiles for {len(test_data)} shops...")
# Select the desired features from the test data
X_test = test_data[selected_features]

# Predict the shop profiles for the test data
y_test_pred = classifier.predict(X_test)

# Add the predicted shop profiles back to the test_data DataFrame
save_data = pd.DataFrame()
save_data['shop_id'] = test_data['shop_id']
save_data['shop_profile'] = y_test_pred
shop_profile_decode = {1:'Low', 2:'Moderate', 3:'High'}
save_data['shop_profile'] = save_data['shop_profile'].replace(shop_profile_decode)

# Save the test_data DataFrame with the predicted shop profiles to a new CSV file
save_data.to_csv('./data/rf_predicted_test_data.csv', index=False)
