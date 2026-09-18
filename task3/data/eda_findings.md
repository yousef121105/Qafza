
EDA Findings Summary — Notebook 4
==================================

1. Top features correlated with is_late:
   - customer_state: late rate ranges from 16.59% (RJ) to 5.43% (PR) across top-volume states (~3x spread)
   - same_state (customer & seller): 5.49% late when same state vs 10.86% when different (~2x difference)
   - primary_seller_state: same signal as same_state, finer granularity, 0 missing values
   - purchase_month: clear seasonal spikes in Nov 2017 (Black Friday, 14.31%) and Feb-Mar 2018 (15.99%, 21.36%)
   - total_freight / total_price: weak standalone correlation (~0.02-0.05), but late orders run ~11-15% higher
     on average, likely useful combined with geography

2. Features needing preprocessing:
   - Right-skewed numeric features (n_payments, total_price, n_distinct_sellers, total_payment_value,
     total_freight) need log-transform or similar
   - Outliers in price/freight/payment fields are not necessarily errors, to be reviewed further
   - Missing values are negligible (<0.02%): order_approved_at (14), order_delivered_carrier_date (1),
     n_payments (1), total_payment_value (1), max_installments (1) - no rows dropped, will impute in NB5
   - purchase_weekday: weak but real signal (~1.7pt spread), low priority

3. Features to exclude:
   - order_status: zero variance (only "delivered" remains after NB2 filtering), no predictive value
   - customer_city: high cardinality (3,747 unique, 32.5% appearing only once), excluded in favor of
     customer_state
   - delivery_days (and anything derived from order_delivered_customer_date): data leakage, not available
     at prediction time, excluded from features entirely

4. Geography decision:
   - Yes, add primary_seller_state and/or same_state in Notebook 5. One of the strongest signals found in
     the entire EDA (~2-3x spread in late rate), full coverage (0 missing). same_state is the cheapest first
     feature to try; primary_seller_state / customer_state can be added as categorical features alongside it.

5. Class imbalance reminder:
   - From Notebook 2: 91.89% on-time vs 8.11% late (~11:1 ratio). Accuracy alone is misleading. Notebook 6
     must use a metric that accounts for the minority class (F1-score, Precision/Recall on is_late=1, or
     PR-AUC), and may need class weighting or resampling during training.
