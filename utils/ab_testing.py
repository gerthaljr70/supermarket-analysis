# %%
import pandas as pd
import numpy as np
import scipy.stats as stats

def run_cross_sell_ab_test(df, target_product, recommended_product, random_state=42):
    """
    Runs a customer-level cross-sell A/B test and evaluates statistical significance.
    
    Parameters:
        df (pd.DataFrame): Main supermarket sales transactions DataFrame.
        target_product (str): Product name bought by the target cohort (e.g., 'Orange Juice 1L').
        recommended_product (str): Recommended cross-sell product (e.g., 'Whole Milk 1L').
        random_state (int): Seed for reproducible random splitting.
        
    Returns:
        pd.DataFrame: Summary table with conversion rates and statistical test results.
    """
    # 1. Filter customers who bought the target product
    target_buyers = df[df['product_name'] == target_product]['customer_id'].unique()
    
    if len(target_buyers) == 0:
        raise ValueError(f"No customers found for target product: '{target_product}'")
        
    # 2. Randomly split target buyers 50/50 into Control and Treatment groups
    np.random.seed(random_state)
    treatment_customers = set(
        np.random.choice(target_buyers, size=len(target_buyers) // 2, replace=False)
    )
    
    # 3. Create experiment DataFrame and assign group labels
    df_exp = df[df['customer_id'].isin(target_buyers)].copy()
    df_exp['ab_group'] = df_exp['customer_id'].apply(
        lambda c: 'Treatment (Banner)' if c in treatment_customers else 'Control (No Banner)'
    )
    
    # 4. Identify customers who bought the recommended product
    rec_buyers = set(df_exp[df_exp['product_name'] == recommended_product]['customer_id'].unique())
    df_exp['bought_recommendation'] = df_exp['customer_id'].isin(rec_buyers)
    
    # 5. Aggregate to 1 row per unique customer first
    customer_summary = df_exp.groupby(['ab_group', 'customer_id'])['bought_recommendation'].first().reset_index()
    
    # 6. Compute metrics per group
    results = customer_summary.groupby('ab_group').agg(
        total_customers=('customer_id', 'nunique'),
        purchased_recommendation=('bought_recommendation', 'sum'),
        conversion_rate=('bought_recommendation', 'mean')
    ).reset_index()
    
    # 7. Dynamically compute not_converted and build contingency table
    results['not_converted'] = results['total_customers'] - results['purchased_recommendation']
    contingency_table = results[['purchased_recommendation', 'not_converted']].values.tolist()
    
    # 8. Run Chi-Square test
    chi2, p_value, dof, _ = stats.chi2_contingency(contingency_table)
    
    # 9. Format output results
    print("=" * 93)
    print(f"A/B TEST REPORT: Target = '{target_product}' -> Recommended ='{recommended_product}'")
    print("=" * 93)
    print(results[['ab_group', 'total_customers', 'purchased_recommendation', 'conversion_rate']].to_string(index=False))
    print("-" * 93)
    print(f"Chi-Square Statistic : {chi2:.4f}")
    print(f"p-value              : {p_value:.4f}")
    print(f"Significance (α=0.05): {'Statistically Significant' if p_value <= 0.05 else 'Not Significant'}")
    print("=" * 93 + "\n")
    
    return results