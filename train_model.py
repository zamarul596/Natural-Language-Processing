"""
Fake News Detection System - Model Training

- Better text preprocessing (keeps important news patterns)
- Larger vocabulary for better generalization
- Balanced training with proper regularization
- Removes dataset-specific bias (Reuters markers.)
"""

import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import numpy as np

print("=" * 60)
print("FAKE NEWS DETECTION - IMPROVED MODEL TRAINING")
print("=" * 60)

# ============================================================
# STEP 1: Load Dataset
# ============================================================
print("\n[1] Loading datasets...")

fake_df = pd.read_csv('data/Fake.csv')
true_df = pd.read_csv('data/True.csv')

print(f"    Fake news articles: {len(fake_df)}")
print(f"    True news articles: {len(true_df)}")

# ============================================================
# STEP 2: Add Labels and Combine
# ============================================================
print("\n[2] Preparing data...")

fake_df['label'] = 0  # 0 = Fake
true_df['label'] = 1  # 1 = Real

# Combine datasets
df = pd.concat([fake_df, true_df], ignore_index=True)

# Shuffle the data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"    Total articles: {len(df)}")

# ============================================================
# STEP 3: Combine title and text
# ============================================================
print("\n[3] Combining title and text...")

df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
df = df[df['content'].str.strip() != '']
print(f"    Valid articles: {len(df)}")

# ============================================================
# STEP 4: Clean Text (IMPROVED - removes dataset bias)
# ============================================================
print("\n[4] Cleaning text (removing dataset bias)...")

def clean_text(text):
    text = str(text)
    
    # Remove Reuters markers (these create bias - real news always has them)
    text = re.sub(r'\(Reuters\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Reuters', '', text, flags=re.IGNORECASE)
    text = re.sub(r'WASHINGTON \(Reuters\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[A-Z]+\s*\(Reuters\)', '', text)
    
    # Remove common news agency markers
    text = re.sub(r'\(AP\)|\(AFP\)|\(CNN\)|\(BBC\)', '', text, flags=re.IGNORECASE)
    
    # Remove location markers at start (e.g., "WASHINGTON -", "NEW YORK (Reuters) -")
    text = re.sub(r'^[A-Z\s,]+\s*[\(\)A-Za-z]*\s*[-–—]\s*', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Keep letters, numbers, and basic punctuation (helps with context)
    text = re.sub(r'[^a-z0-9\s\.\,\!\?]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

df['clean_text'] = df['content'].apply(clean_text)
df = df[df['clean_text'].str.len() > 100]  # Need more context
print(f"    After cleaning: {len(df)}")

# Show sample
print("\n    Sample cleaned text:")
print(f"    '{df['clean_text'].iloc[0][:100]}...'")

# ============================================================
# STEP 5: Prepare Features
# ============================================================
print("\n[5] Preparing features...")

X = df['clean_text'].values
y = df['label'].values

print(f"    Features: {len(X)}")
print(f"    Labels - Fake: {sum(y==0)}, Real: {sum(y==1)}")

# ============================================================
# STEP 6: Split Data
# ============================================================
print("\n[6] Splitting data (80/20)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"    Training: {len(X_train)}")
print(f"    Testing: {len(X_test)}")

# ============================================================
# STEP 7: TF-IDF Vectorization
# ============================================================
print("\n[7] Creating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=10000,      # More features for better generalization
    ngram_range=(1, 3),      # Include trigrams for better context
    min_df=3,                # Minimum document frequency
    max_df=0.95,             # Remove very common words
    sublinear_tf=True,       # Apply sublinear tf scaling
    stop_words='english'
)

# Fit and transform training data
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"    TF-IDF features: {X_train_tfidf.shape[1]}")
print(f"    Vectorizer fitted: {hasattr(vectorizer, 'idf_')}")

# ============================================================
# STEP 8: Train Model (IMPROVED - better regularization)
# ============================================================
print("\n[8] Training Logistic Regression (improved)...")

model = LogisticRegression(
    max_iter=2000,
    random_state=42,
    C=0.5,                   # Stronger regularization to prevent overfitting
    class_weight='balanced', # Handle any class imbalance
    solver='lbfgs'
)
model.fit(X_train_tfidf, y_train)

print("    Model trained!")

# ============================================================
# STEP 9: Evaluate Model
# ============================================================
print("\n[9] Evaluating model...")

y_pred = model.predict(X_test_tfidf)
y_prob = model.predict_proba(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n    ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("\n    Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

# ============================================================
# STEP 10: Save Model
# ============================================================
print("\n[10] Saving model...")

os.makedirs('model', exist_ok=True)

# Save using pickle
model_data = {
    'model': model,
    'vectorizer': vectorizer,
    'accuracy': accuracy
}

with open('model/fake_news_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("    Model saved to: model/fake_news_model.pkl")

# ============================================================
# STEP 11: Verify Saved Model
# ============================================================
print("\n[11] Verifying saved model...")

with open('model/fake_news_model.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

loaded_vectorizer = loaded_data['vectorizer']
loaded_model = loaded_data['model']

print(f"    Vectorizer has idf_: {hasattr(loaded_vectorizer, 'idf_')}")

# ============================================================
# STEP 12: Test with Real-World Examples
# ============================================================
print("\n[12] Testing with real-world examples...")

test_cases = [
    ("FAKE", "BREAKING: Scientists prove that 5G towers cause mind control! Share before they delete this!"),
    ("REAL", "The Senate passed a $1.2 trillion infrastructure bill on Tuesday with bipartisan support, sending the measure to President Biden's desk."),
    ("REAL", "Apple Inc. reported quarterly revenue of $89.6 billion, an increase of 36 percent from the year-ago quarter."),
    ("FAKE", "EXPOSED: Secret government documents reveal aliens have been living among us since 1947!"),
]

for expected, text in test_cases:
    clean = clean_text(text)
    tfidf = loaded_vectorizer.transform([clean])
    pred = loaded_model.predict(tfidf)[0]
    prob = loaded_model.predict_proba(tfidf)[0]
    result = "REAL" if pred == 1 else "FAKE"
    confidence = max(prob) * 100
    status = "✓" if result == expected else "✗"
    print(f"    {status} Expected: {expected}, Got: {result} ({confidence:.1f}%)")

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print(f"Accuracy: {accuracy*100:.2f}%")
print("=" * 60)
print("\nTo run the web app:")
print("  .\\venv\\Scripts\\python.exe app.py")
print("=" * 60)
