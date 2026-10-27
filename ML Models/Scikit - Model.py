# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# ==============================================================================
# 1. LOAD DATASET
# ==============================================================================
df = pd.read_csv('./dados/supermarket_sales.csv')

# %%
# ==============================================================================
# 2. FEATURE ENGINEERING (AGGREGATE TO CUSTOMER-LEVEL)
# ==============================================================================
# Define target product
target_product = 'Whole Milk 1L'

# Find customer IDs who bought Whole Milk 1L
target_buyers = set(df[df['product_name'] == target_product]['customer_id'].unique())

# Create additional feature flags (e.g., whether customer bought Orange Juice)
oj_buyers = set(df[df['product_name'] == 'Orange Juice 1L']['customer_id'].unique())

# Aggregate transaction data to 1 row per unique customer
customer_df = df.groupby('customer_id').agg(
    total_baskets=('basket_id', 'nunique'),
    total_items=('product_sku', 'count'),
    unique_products=('product_name', 'nunique')
).reset_index()

# Add binary features and target column (1 = Yes, 0 = No)
customer_df['bought_orange_juice'] = customer_df['customer_id'].isin(oj_buyers).astype(int)
customer_df['target'] = customer_df['customer_id'].isin(target_buyers).astype(int)

# ==============================================================================
# 3. SEPARATE FEATURES (X) AND TARGET (y)
# ==============================================================================
feature_cols = ['total_baskets', 'total_items', 'unique_products', 'bought_orange_juice']

X = customer_df[feature_cols]
y = customer_df['target']

# ==============================================================================
# 4. TRAIN-TEST SPLIT & FEATURE SCALING
# ==============================================================================
# Split into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Scale features so all variables have mean=0 and variance=1
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================================================================
# 5. TRAIN MODEL & PREDICT
# ==============================================================================
# Initialize and train Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions on test set
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# ==============================================================================
# 6. EVALUATE MODEL PERFORMANCE
# ==============================================================================
print("=" * 53)
print("SUPERMARKET CUSTOMER CLASSIFICATION REPORT")
print("=" * 53)
print(classification_report(y_test, y_pred))
print("-" * 53)
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
print("=" * 53)
# %%
