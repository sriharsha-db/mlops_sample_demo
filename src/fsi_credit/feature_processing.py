from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql.functions import *
from pyspark.sql import SparkSession
from datetime import date
from argparse import ArgumentParser


def main():
    parser = ArgumentParser(description='parse the command line arguments for fsi feature pre processing')
    parser.add_argument('--uc_catalog', type=str, required=True)
    parser.add_argument('--uc_schema', type=str, required=True)
    input_args = parser.parse_args()
    catalog_name = input_args.uc_catalog
    schema_name = input_args.uc_schema

    print(f"input parsed details catalog:{catalog_name} and schema:{schema_name}")

    spark = SparkSession.builder \
        .appName("fsi_feature_processing") \
        .getOrCreate()

    fe = FeatureEngineeringClient()

    customer_gold_features = (spark.table(f"{catalog_name}.{schema_name}.customer_gold_source")
                              .withColumn('age', int(date.today().year) - col('birth_year'))
                              .select('cust_id', 'education', 'marital_status', 'months_current_address',
                                      'months_employment', 'is_resident',
                                      'tenure_months', 'product_cnt', 'tot_rel_bal', 'revenue_tot', 'revenue_12m',
                                      'income_annual', 'tot_assets',
                                      'overdraft_balance_amount', 'overdraft_number', 'total_deposits_number',
                                      'total_deposits_amount', 'total_equity_amount',
                                      'total_UT', 'customer_revenue', 'age', 'avg_balance', 'num_accs', 'balance_usd',
                                      'available_balance_usd')).dropDuplicates(['cust_id'])

    telco_gold_features = (spark.table(f"{catalog_name}.{schema_name}.telco_gold_source")
                           .select('cust_id', 'is_pre_paid', 'number_payment_delays_last12mo',
                                   'pct_increase_annual_number_of_delays_last_3_year', 'phone_bill_amt',
                                   'avg_phone_bill_amt_lst12mo')).dropDuplicates(['cust_id'])

    fund_trans_gold_features = spark.table(f"{catalog_name}.{schema_name}.fund_trans_gold_source").dropDuplicates(
        ['cust_id'])

    for c in ['12m', '6m', '3m']:
        fund_trans_gold_features = fund_trans_gold_features.withColumn('tot_txn_cnt_' + c,
                                                                       col('sent_txn_cnt_' + c) + col('rcvd_txn_cnt_' + c)) \
            .withColumn('tot_txn_amt_' + c, col('sent_txn_amt_' + c) + col('rcvd_txn_amt_' + c))

    fund_trans_gold_features = fund_trans_gold_features.withColumn('ratio_txn_amt_3m_12m',
                                                                   when(col('tot_txn_amt_12m') == 0, 0).otherwise(
                                                                       col('tot_txn_amt_3m') / col('tot_txn_amt_12m'))) \
        .withColumn('ratio_txn_amt_6m_12m',
                    when(col('tot_txn_amt_12m') == 0, 0).otherwise(col('tot_txn_amt_6m') / col('tot_txn_amt_12m'))) \
        .na.fill(0)

    feature_df = customer_gold_features.join(telco_gold_features.alias('telco'), "cust_id", how="left")
    feature_df = feature_df.join(fund_trans_gold_features, "cust_id", how="left")

    fe.write_table(
      df=feature_df,
      name=f"{catalog_name}.{schema_name}.credit_decision_fs",
      mode='merge'
    )


if __name__ == "__main__":
    main()