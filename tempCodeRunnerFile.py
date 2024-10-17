ordered_model = CatBoostRegressor(cat_features=cat_features, verbose=0, random_state=42, boosting_type='Ordered')
ordered_scores = cross_val_score(ordered_model, X, t, cv=5, scoring='r2')
print(f"Average R² score for ordered CatBoost: {ordered_scores.mean():.4f}")