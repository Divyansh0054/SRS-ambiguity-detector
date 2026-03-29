import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import joblib

# 1. Load dataset
data = pd.read_csv("data/requirement.csv")

X = data["requirement"]
y = data["label"]

# 2. Train-test split (with stratification)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. TF-IDF Vectorization (IMPROVED 🔥)
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),   # captures phrases
    max_features=5000     # Captures phrases like "user friendly" → better accuracy
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Train classifier (IMPROVED 🔥)
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"   # handles class imbalance
)

model.fit(X_train_tfidf, y_train)

# 5. Predictions
y_pred = model.predict(X_test_tfidf)
y_prob = model.predict_proba(X_test_tfidf)  # for confidence (future UI)

# 6. Evaluation (VERY IMPORTANT 🔥)
print("\n📊 MODEL EVALUATION")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

print("\nDetailed Classification Report:\n")
print(classification_report(y_test, y_pred))

# Show class mapping
print("Classes:", model.classes_)  # 0 = Clear, 1 = Ambiguous

# 7. Save model and vectorizer
joblib.dump(model, "ambiguity_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\n✅ Model and vectorizer saved successfully")

# Note:
# Ambiguity is subjective, so extremely high accuracy is not expected.
# Focus is on explainability (rule-based) + ML support.