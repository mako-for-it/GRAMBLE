import pandas as pd

df = pd.read_csv("outputs/xh/evaluation_results.csv")

avg_bleu = df['bleu score'].mean()
avg_chrf = df['chrf'].mean()

max_bleu = df['bleu score'].max()
max_chrf = df['chrf'].max()

best_bleu = df.loc[df['bleu score'].idxmax()]
best_chrf = df.loc[df['chrf'].idxmax()]

median_bleu = df['bleu score'].median()
median_chrf = df['chrf'].median()

print(f"avg bleu is {avg_bleu}")
print(f"avg chrf is {avg_chrf}")
print("")
print(f"median chrf is {median_chrf}")
print(f"max chrf is {max_chrf} with ")
print(f"median bleu is {median_bleu}")
print(f"max bleu is {max_bleu} with ")