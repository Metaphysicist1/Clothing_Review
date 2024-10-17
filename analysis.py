import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor
from main import X, y, cat_features
import seaborn as sns
import numpy as np
import pandas as pd


kf = KFold(n_splits=5)
feature_importances = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model = CatBoostRegressor(cat_features=cat_features, verbose=0, random_state=42)
    model.fit(X_train, y_train)
    feature_importances.append(model.get_feature_importance())

avg_importance = np.mean(feature_importances, axis=0)

feat_imp_df = pd.DataFrame({'feature': X.columns,'importance': avg_importance})

top_features = feat_imp_df.sort_values(by='importance', ascending=False).head(20)

plt.figure(figsize=(12,10))
ax = sns.barplot(x='importance', y='feature',data=top_features)

plt.title('Top 20 Most Important Features - CatBoost Model',fontsize=20, fontweight='bold')
plt.xlabel('Importance Score', fontsize=15)
plt.ylabel('Feature', fontsize=15)

plt.tight_layout()
plt.show() 