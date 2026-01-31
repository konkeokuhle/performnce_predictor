import pandas as pd

df = pd.read_csv("data/student_performance.csv")

print("\n--- DATA INFO ---")
print(df.info())

print("\n--- STATISTICS ---")
print(df.describe())

print("\n--- AT RISK DISTRIBUTION ---")
print(df["at_risk"].value_counts())
