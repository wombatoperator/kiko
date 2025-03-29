import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import argparse
import os

def main(input_file, model_dir):
    """Main function to train the model."""
    # Create model directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    print(f"Reading cleaned data from {input_file}")
    df = pd.read_csv(input_file)
    
    # Convert label to binary if needed (assuming threshold of 0.5)
    df['binary_label'] = (df['label'] >= 0.5).astype(int)
    
    # Split the data
    print("Splitting data into train and test sets")
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'], df['binary_label'], test_size=0.2, random_state=42
    )
    
    # Vectorize the text
    print("Vectorizing text data")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train the model
    print("Training the model")
    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X_train_vec, y_train)
    
    # Evaluate the model
    print("Evaluating the model")
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model and vectorizer
    print(f"Saving model to {model_dir}")
    joblib.dump(model, os.path.join(model_dir, 'site_classifier_model.joblib'))
    joblib.dump(vectorizer, os.path.join(model_dir, 'tfidf_vectorizer.joblib'))
    
    print("Model training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train site classifier model")
    parser.add_argument("--input", type=str, default="data/processed/cleaned_site_data.csv",
                        help="Path to cleaned CSV file")
    parser.add_argument("--model-dir", type=str, default="models",
                        help="Directory to save the trained model")
    
    args = parser.parse_args()
    main(args.input, args.model_dir)
