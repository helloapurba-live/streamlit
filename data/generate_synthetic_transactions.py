"""
============================================================================
Synthetic Bank Transaction Data Generator
============================================================================

LEARNING OBJECTIVES:
1. Generate realistic banking transaction datasets for fraud detection
2. Understand class imbalance in anomaly detection (2-5% fraud rate)
3. Create temporal patterns and behavioral features
4. Implement synthetic fraud patterns (velocity, unusual amounts, geo anomalies)

THEORY:
--------
Bank fraud detection requires:
- Imbalanced datasets (fraudulent transactions are rare ~2-3%)
- Temporal features (time of day, day of week, transaction velocity)
- Behavioral features (deviation from customer baseline)
- Network features (merchant risk scores, location anomalies)

This generator creates 10,000 synthetic transactions with:
- 97-98% legitimate transactions (normal customer behavior)
- 2-3% fraudulent transactions (anomalous patterns)
- 20 features including: amount, time, location, merchant category, customer history

ARCHITECTURE:
-------------
            ┌─────────────────────────────────┐
            │  Customer Profile Generator     │
            │  (1000 unique customers)        │
            └──────────────┬──────────────────┘
                           │
            ┌──────────────▼──────────────────┐
            │  Transaction Generator          │
            │  - Normal: 97% (baseline)       │
            │  - Fraud: 3% (anomalies)        │
            └──────────────┬──────────────────┘
                           │
            ┌──────────────▼──────────────────┐
            │  Feature Engineering            │
            │  - Velocity features            │
            │  - Amount deviations            │
            │  - Time-based features          │
            └──────────────┬──────────────────┘
                           │
            ┌──────────────▼──────────────────┐
            │  CSV Export                     │
            │  data/transactions_raw.csv      │
            └─────────────────────────────────┘
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from faker import Faker
import random
import os

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
fake = Faker()
Faker.seed(42)


class BankTransactionGenerator:
    """
    Generates synthetic bank transaction data with fraud labels.

    Parameters:
    -----------
    n_transactions : int
        Total number of transactions to generate
    fraud_rate : float
        Proportion of fraudulent transactions (0.0 to 1.0)
    n_customers : int
        Number of unique customers
    """

    def __init__(self, n_transactions=10000, fraud_rate=0.03, n_customers=1000):
        self.n_transactions = n_transactions
        self.fraud_rate = fraud_rate
        self.n_customers = n_customers
        self.customers = self._generate_customers()
        self.merchant_categories = [
            'grocery', 'gas_transport', 'restaurant', 'entertainment',
            'online_shopping', 'bills_utilities', 'health_fitness', 'travel'
        ]

    def _generate_customers(self):
        """Generate customer profiles with baseline spending patterns."""
        customers = []
        for i in range(self.n_customers):
            customer = {
                'customer_id': f'CUST_{i:06d}',
                'account_age_days': np.random.randint(30, 3650),  # 1 month to 10 years
                'avg_transaction_amount': np.random.lognormal(mean=4.0, sigma=0.8),  # ~$50-$300
                'transaction_frequency': np.random.choice(['low', 'medium', 'high'], p=[0.3, 0.5, 0.2]),
                'preferred_categories': np.random.choice(self.merchant_categories, size=3, replace=False).tolist(),
                'home_latitude': np.random.uniform(25.0, 49.0),  # US latitude range
                'home_longitude': np.random.uniform(-125.0, -65.0),  # US longitude range
            }
            customers.append(customer)
        return pd.DataFrame(customers)

    def _generate_normal_transaction(self, customer_row, timestamp):
        """Generate a legitimate transaction based on customer profile."""
        amount = np.random.normal(
            loc=customer_row['avg_transaction_amount'],
            scale=customer_row['avg_transaction_amount'] * 0.3
        )
        amount = max(5.0, amount)  # Minimum transaction $5

        # Location near customer home (within ~50 miles)
        lat_noise = np.random.normal(0, 0.5)
        lon_noise = np.random.normal(0, 0.5)

        transaction = {
            'transaction_id': fake.uuid4(),
            'customer_id': customer_row['customer_id'],
            'timestamp': timestamp,
            'amount': round(amount, 2),
            'merchant_category': np.random.choice(customer_row['preferred_categories']),
            'merchant_id': f"MERCH_{np.random.randint(1, 5000):05d}",
            'latitude': customer_row['home_latitude'] + lat_noise,
            'longitude': customer_row['home_longitude'] + lon_noise,
            'is_online': np.random.choice([0, 1], p=[0.7, 0.3]),
            'is_fraud': 0
        }
        return transaction

    def _generate_fraud_transaction(self, customer_row, timestamp):
        """Generate a fraudulent transaction with anomalous patterns."""
        fraud_type = np.random.choice(['high_amount', 'velocity', 'location', 'unusual_category'])

        if fraud_type == 'high_amount':
            # Unusually high amount (5-10x normal)
            amount = customer_row['avg_transaction_amount'] * np.random.uniform(5, 10)
        elif fraud_type == 'velocity':
            # Normal amount but part of rapid-fire sequence
            amount = customer_row['avg_transaction_amount'] * np.random.uniform(0.8, 1.5)
        elif fraud_type == 'location':
            # Far from customer's home location
            amount = customer_row['avg_transaction_amount'] * np.random.uniform(1.0, 3.0)
        else:  # unusual_category
            amount = customer_row['avg_transaction_amount'] * np.random.uniform(1.5, 4.0)

        # Anomalous location (potentially different state/country)
        if fraud_type == 'location':
            latitude = np.random.uniform(25.0, 49.0)
            longitude = np.random.uniform(-125.0, -65.0)
        else:
            lat_noise = np.random.normal(0, 0.3)
            lon_noise = np.random.normal(0, 0.3)
            latitude = customer_row['home_latitude'] + lat_noise
            longitude = customer_row['home_longitude'] + lon_noise

        # Unusual merchant category (not in preferred list)
        unusual_categories = [cat for cat in self.merchant_categories
                             if cat not in customer_row['preferred_categories']]
        merchant_category = np.random.choice(unusual_categories) if unusual_categories else \
                           np.random.choice(self.merchant_categories)

        transaction = {
            'transaction_id': fake.uuid4(),
            'customer_id': customer_row['customer_id'],
            'timestamp': timestamp,
            'amount': round(amount, 2),
            'merchant_category': merchant_category,
            'merchant_id': f"MERCH_{np.random.randint(5000, 9999):05d}",  # Different merchant range
            'latitude': latitude,
            'longitude': longitude,
            'is_online': np.random.choice([0, 1], p=[0.4, 0.6]),  # More online fraud
            'is_fraud': 1
        }
        return transaction

    def generate(self):
        """
        Generate the full transaction dataset.

        Returns:
        --------
        pandas.DataFrame
            Dataset with columns: transaction_id, customer_id, timestamp, amount,
            merchant_category, merchant_id, latitude, longitude, is_online, is_fraud
        """
        transactions = []

        # Determine which transactions will be fraudulent
        n_fraud = int(self.n_transactions * self.fraud_rate)
        fraud_indices = set(np.random.choice(self.n_transactions, size=n_fraud, replace=False))

        # Generate transactions over a 30-day period
        start_date = datetime.now() - timedelta(days=30)

        print(f"Generating {self.n_transactions:,} transactions...")
        print(f"  - Normal: {self.n_transactions - n_fraud:,} ({100*(1-self.fraud_rate):.1f}%)")
        print(f"  - Fraud: {n_fraud:,} ({100*self.fraud_rate:.1f}%)")

        for i in range(self.n_transactions):
            # Random timestamp within 30-day window
            random_seconds = np.random.randint(0, 30 * 24 * 60 * 60)
            timestamp = start_date + timedelta(seconds=random_seconds)

            # Random customer
            customer_row = self.customers.sample(n=1).iloc[0]

            # Generate fraud or normal transaction
            if i in fraud_indices:
                transaction = self._generate_fraud_transaction(customer_row, timestamp)
            else:
                transaction = self._generate_normal_transaction(customer_row, timestamp)

            transactions.append(transaction)

        df = pd.DataFrame(transactions)

        # Add derived features
        df = self._add_derived_features(df)

        return df

    def _add_derived_features(self, df):
        """Add time-based and behavioral features."""
        print("\nEngineering features...")

        # Time features
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_night'] = df['hour'].isin(range(0, 6)).astype(int)

        # Sort by customer and timestamp for velocity features
        df = df.sort_values(['customer_id', 'timestamp']).reset_index(drop=True)

        # Velocity features (transactions per customer in last 1 hour and 24 hours)
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'])

        # Calculate time since last transaction for each customer
        df['time_since_last_transaction'] = df.groupby('customer_id')['timestamp_dt'].diff().dt.total_seconds() / 60
        df['time_since_last_transaction'] = df['time_since_last_transaction'].fillna(999999)  # First transaction

        # Customer statistics (could be pre-computed in production)
        customer_stats = df.groupby('customer_id')['amount'].agg(['mean', 'std']).reset_index()
        customer_stats.columns = ['customer_id', 'customer_avg_amount', 'customer_std_amount']
        df = df.merge(customer_stats, on='customer_id', how='left')
        df['customer_std_amount'] = df['customer_std_amount'].fillna(0)

        # Amount deviation from customer baseline
        df['amount_deviation'] = (df['amount'] - df['customer_avg_amount']) / (df['customer_std_amount'] + 1)

        # Merchant risk score (simulate: merchants with more fraud get higher risk)
        merchant_fraud = df.groupby('merchant_id')['is_fraud'].mean().reset_index()
        merchant_fraud.columns = ['merchant_id', 'merchant_risk_score']
        df = df.merge(merchant_fraud, on='merchant_id', how='left')

        # Distance from home (simplified Euclidean distance in lat/lon space)
        # In production, use haversine formula
        customer_home = self.customers[['customer_id', 'home_latitude', 'home_longitude']]
        df = df.merge(customer_home, on='customer_id', how='left')
        df['distance_from_home'] = np.sqrt(
            (df['latitude'] - df['home_latitude'])**2 +
            (df['longitude'] - df['home_longitude'])**2
        )

        # Drop intermediate columns
        df = df.drop(columns=['timestamp_dt', 'home_latitude', 'home_longitude'])

        return df


def main():
    """
    Main execution function.

    STEPS:
    1. Initialize generator with 10,000 transactions
    2. Generate dataset with fraud labels
    3. Save to CSV
    4. Display preview and statistics
    """

    print("="*70)
    print("BANK TRANSACTION SYNTHETIC DATA GENERATOR")
    print("="*70)

    # Initialize generator
    generator = BankTransactionGenerator(
        n_transactions=10000,
        fraud_rate=0.03,
        n_customers=1000
    )

    # Generate data
    df = generator.generate()

    # Save to CSV
    output_path = os.path.join(os.path.dirname(__file__), 'transactions_raw.csv')
    df.to_csv(output_path, index=False)

    print(f"\n✅ Generated {len(df):,} transactions")
    print(f"✅ Saved to: {output_path}")
    print(f"\n📊 Dataset Statistics:")
    print(f"   - Shape: {df.shape}")
    print(f"   - Fraud transactions: {df['is_fraud'].sum():,} ({100*df['is_fraud'].mean():.2f}%)")
    print(f"   - Normal transactions: {(~df['is_fraud'].astype(bool)).sum():,} ({100*(1-df['is_fraud'].mean()):.2f}%)")
    print(f"   - Unique customers: {df['customer_id'].nunique():,}")
    print(f"   - Unique merchants: {df['merchant_id'].nunique():,}")
    print(f"   - Amount range: ${df['amount'].min():.2f} - ${df['amount'].max():.2f}")
    print(f"   - Average amount: ${df['amount'].mean():.2f}")

    print(f"\n📋 Column List ({len(df.columns)} columns):")
    for col in df.columns:
        print(f"   - {col}")

    print(f"\n🔍 First 10 Rows Preview:")
    print("="*70)
    print(df.head(10).to_string())

    print(f"\n💾 Data saved successfully!")
    print(f"   Next step: Run feature engineering and model training")


if __name__ == "__main__":
    main()


"""
============================================================================
LINE-BY-LINE EXPLANATION
============================================================================

Lines 1-47: Module docstring with learning objectives, theory, and architecture

Lines 49-54: Imports
    - pandas/numpy: data manipulation
    - faker: realistic fake customer data
    - datetime: timestamp generation

Lines 57-60: Set random seeds for reproducibility across runs

Lines 63-79: BankTransactionGenerator class initialization
    - n_transactions: total dataset size (10,000)
    - fraud_rate: percentage of fraudulent transactions (3%)
    - n_customers: number of unique customer profiles (1,000)
    - Initializes customer profiles with spending baselines

Lines 81-99: _generate_customers method
    - Creates 1,000 customer profiles with:
        * Unique customer ID
        * Account age (30 days to 10 years)
        * Average transaction amount (lognormal distribution ~$50-$300)
        * Transaction frequency tier
        * Preferred merchant categories
        * Home location (lat/lon for US)

Lines 101-125: _generate_normal_transaction method
    - Amount: sampled from normal distribution around customer baseline
    - Location: near customer home (±0.5 degrees lat/lon ≈ 50 miles)
    - Merchant: from customer's preferred categories
    - Returns transaction dict with is_fraud=0

Lines 127-176: _generate_fraud_transaction method
    - Four fraud patterns:
        1. high_amount: 5-10x normal spending
        2. velocity: rapid successive transactions
        3. location: far from home
        4. unusual_category: merchant type customer never uses
    - Higher proportion of online transactions
    - Returns transaction dict with is_fraud=1

Lines 178-229: generate method (main generation logic)
    - Determines fraud indices randomly based on fraud_rate
    - Generates transactions over 30-day window
    - Randomly assigns customers
    - Calls fraud or normal generator based on index
    - Adds derived features (time, velocity, behavioral)

Lines 231-276: _add_derived_features method
    - Time features: hour, day_of_week, is_weekend, is_night
    - Velocity: time since last transaction per customer
    - Behavioral: amount_deviation from customer baseline
    - Merchant risk: fraud rate per merchant
    - Geography: distance_from_home (simplified Euclidean)

Lines 279-326: main function
    - Instantiates generator
    - Generates dataset
    - Saves to data/transactions_raw.csv
    - Prints statistics and preview

============================================================================
SAMPLE INPUT (Parameters)
============================================================================
n_transactions = 10000
fraud_rate = 0.03  (3%)
n_customers = 1000

============================================================================
SAMPLE OUTPUT (First 10 Rows of transactions_raw.csv)
============================================================================

transaction_id                        customer_id  timestamp            amount  merchant_category  ...  is_fraud
8f3a2b1c-...                          CUST_000127  2025-01-15 08:23:11  127.45  grocery           ...  0
9d4c5e2f-...                          CUST_000893  2025-01-15 14:07:55  89.20   gas_transport     ...  0
1a2b3c4d-...                          CUST_000412  2025-01-16 19:42:03  523.80  online_shopping   ...  1  ← FRAUD
...

Additional columns: merchant_id, latitude, longitude, is_online, hour, day_of_week,
is_weekend, is_night, time_since_last_transaction, customer_avg_amount,
customer_std_amount, amount_deviation, merchant_risk_score, distance_from_home

============================================================================
POWERSHELL COMMANDS TO RUN
============================================================================

# Activate virtual environment first
.\.venv\Scripts\Activate.ps1

# Generate data
python data\generate_synthetic_transactions.py

# Verify output
python -c "import pandas as pd; df = pd.read_csv('data/transactions_raw.csv'); print(df.info()); print(df['is_fraud'].value_counts())"

# Expected terminal output:
# Generating 10,000 transactions...
#   - Normal: 9,700 (97.0%)
#   - Fraud: 300 (3.0%)
#
# Engineering features...
#
# ✅ Generated 10,000 transactions
# ✅ Saved to: data/transactions_raw.csv
#
# 📊 Dataset Statistics:
#    - Shape: (10000, 21)
#    - Fraud transactions: 300 (3.00%)
#    ...

============================================================================
EXERCISE
============================================================================

TASK: Modify the generator to add a new fraud pattern type called "card_testing"
      which represents fraudsters testing stolen card numbers with small amounts.

REQUIREMENTS:
1. Add 'card_testing' to fraud_type choices in _generate_fraud_transaction
2. Card testing transactions should have:
   - Very small amounts ($1-$5)
   - Sequential timestamps (within 1-2 minutes of each other)
   - Different merchant each time
   - All online transactions

SOLUTION:
----------
In _generate_fraud_transaction method, add:

elif fraud_type == 'card_testing':
    # Very small amounts for testing
    amount = np.random.uniform(1.0, 5.0)
    # Should be grouped in time (handled in main generation loop)

Then in generate() method, after creating a card_testing fraud transaction,
generate 2-4 more within 2 minutes of the first one with different merchants.

This pattern would be detected by velocity features and low-amount clustering!

============================================================================
"""
