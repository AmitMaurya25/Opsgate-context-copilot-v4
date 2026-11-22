import pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
d = pd.read_csv(Path(__file__).parent / "results.csv")
fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
d.groupby(["category", "mode"]).input_tokens.mean().unstack().plot.bar(ax=ax[0], title="Avg context tokens")
d.groupby(["category", "mode"]).correct.mean().unstack().plot.bar(ax=ax[1], title="Accuracy", ylim=(0, 1))
d.groupby(["category", "mode"]).effective_tokens.mean().unstack().plot.bar(ax=ax[2], title="Avg billed-equivalent tokens (sent minus KV-cache hits)")
plt.tight_layout(); plt.savefig(Path(__file__).parent / "results.png", dpi=150)
print(d.groupby("mode")[["input_tokens", "sent_tokens", "cache_hit_pct", "effective_tokens", "correct"]].mean())
