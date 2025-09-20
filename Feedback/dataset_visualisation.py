import pandas as pd

csv_path = "arxiv_sample_with_predictions.csv"
final_df = pd.read_csv(csv_path)
print(final_df.head())