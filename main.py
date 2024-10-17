from catboost import CatBoostRegressor
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score


df = pd.read_csv('data.csv')
df = df.fillna('missing')

cols = ['Clothing ID', 'Age', 'Rating', 'Recommended IND', 'Positive Feedback Count', 'Division Name','Department Name', 'Class Name']

df = df[cols]

X = df.drop('Rating',axis=1)
y = df['Rating']

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

cat_features = X.select_dtypes(include=['object']).columns.tolist()


print(cat_features)

model = CatBoostRegressor(cat_features=cat_features, verbose=0, random_state=42)
default_scores = cross_val_score(model, train_X, train_y, cv=5, scoring='r2')
print(f"Average R² score for default CatBoost: {default_scores.mean():.4f}")

ordered_model = CatBoostRegressor(cat_features=cat_features, verbose=0, random_state=42, boosting_type='Ordered')
ordered_scores = cross_val_score(ordered_model, train_X, train_y, cv=5, scoring='r2')
print(f"Average R² score for ordered CatBoost: {ordered_scores.mean():.4f}")

print("Start Training")

model.fit(train_X, train_y)
ordered_model.fit(train_X, train_y)

print("Training Complete")

print(f"Default CatBoost R² score on test set: {model.score(test_X, test_y):.4f}")
print(f"Ordered CatBoost R² score on test set: {ordered_model.score(test_X, test_y):.4f}")

model.save_model('catboost.cbm')
ordered_model.save_model('ordered_catboost.cbm')