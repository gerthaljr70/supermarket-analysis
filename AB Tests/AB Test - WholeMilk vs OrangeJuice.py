# %%
import pandas as pd
from utils.ab_testing import run_cross_sell_ab_test

# Load your data first
df = pd.read_csv('./dados/supermarket_sales.csv')

# Test Whole Milk buyers -> Orange Juice recommendation
run_cross_sell_ab_test(df, target_product='Whole Milk 1L', recommended_product='Orange Juice 1L')
