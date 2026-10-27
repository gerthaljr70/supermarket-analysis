# %%
import pandas as pd
from utils.ab_testing import run_cross_sell_ab_test

# Load your data first
df = pd.read_csv('./dados/supermarket_sales.csv')

# Test Orange Juice buyers -> Whole Milk recommendation
run_cross_sell_ab_test(df, target_product='Orange Juice 1L', recommended_product='Whole Milk 1L')

# %%
