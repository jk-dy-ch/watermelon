import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, f1_score
np.random.seed(42)
x = np.random.rand(100)
y = 2 * x + np.random.normal(0, 0.1, 100)
r, p = stats.pearsonr(x, y)
print(f"相关系数(R值): {r:.4f}")
print(f"P值: {p:.4f}")
y_true_binary = (y > np.median(y)).astype(int)
y_scores = y
auc = roc_auc_score(y_true_binary, y_scores)
print(f"\nAUC值: {auc:.4f}")
y_pred_binary = (y_scores > np.median(y_scores)).astype(int)
f1 = f1_score(y_true_binary, y_pred_binary)
print(f"\nF1值: {f1:.4f}")