# Entity Resolution Project - Banking Use Case
# Author: [Your Name]
# Description: End-to-end entity resolution pipeline with Streamlit UI, including rule-based and ML-based matching

# === MODULES ===
import pandas as pd
import numpy as np
import streamlit as st
from thefuzz import fuzz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

# === STEP 1: DATA INGESTION ===
def load_data():
    st.sidebar.subheader("Upload Input Files (Optional)")
    uploaded_file1 = st.sidebar.file_uploader("Upload Source File 1 (CSV)", type=["csv"], key="file1")
    uploaded_file2 = st.sidebar.file_uploader("Upload Source File 2 (CSV)", type=["csv"], key="file2")

    if uploaded_file1 is not None and uploaded_file2 is not None:
        df1 = pd.read_csv(uploaded_file1)
        df2 = pd.read_csv(uploaded_file2)
        st.sidebar.success("Using uploaded files.")
    else:
        st.sidebar.info("Using default source1.csv and source2.csv from data/raw/")
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/raw'))
        df1 = pd.read_csv(os.path.join(base_path, "source1.csv"))
        df2 = pd.read_csv(os.path.join(base_path, "source2.csv"))

    return df1, df2

# === STEP 2: BLOCKING ===
def blocking(df1, df2):
    df1["block_key"] = df1["last_name"].str[0] + df1["zip"].astype(str)
    df2["block_key"] = df2["last_name"].str[0] + df2["zip"].astype(str)
    return pd.merge(df1, df2, on="block_key", suffixes=("_1", "_2"))

# === STEP 3: FEATURE GENERATION ===
def compute_features(df):
    df["name_sim"] = df.apply(lambda x: fuzz.token_sort_ratio(x["full_name_1"], x["full_name_2"]), axis=1)
    df["addr_sim"] = df.apply(lambda x: fuzz.token_sort_ratio(x["address_1"], x["address_2"]), axis=1)
    df["email_sim"] = df.apply(lambda x: fuzz.partial_ratio(str(x["email_1"]), str(x["email_2"])), axis=1)
    return df[["name_sim", "addr_sim", "email_sim"]], df

# === STEP 4: MATCHING ===
def rule_based_match(df, threshold):
    df["score"] = (df["name_sim"]*0.4 + df["addr_sim"]*0.4 + df["email_sim"]*0.2)
    df["is_match"] = df["score"] > threshold
    return df

def ml_based_match(df, labels):
    X, _ = compute_features(df)
    y = labels

    if len(df) == 0:
        st.warning("No data available for matching.")
        return pd.DataFrame()

    if len(np.unique(y)) < 2:
        st.warning("Not enough class diversity to train ML model. Try with more data.")
        return pd.DataFrame()

    if len(y) < 5:
        st.warning("Too few samples to split into train and test. Try with more data.")
        return pd.DataFrame()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]

    df_test = df.iloc[X_test.index].copy()
    df_test["ml_is_match"] = preds
    df_test["match_confidence"] = probs

    st.write("Precision:", precision_score(y_test, preds, zero_division=0))
    st.write("Recall:", recall_score(y_test, preds, zero_division=0))

    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Match", "Match"], yticklabels=["Non-Match", "Match"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    st.subheader("Manual Review - Top 10 Most Uncertain Predictions")
    review_set = df_test[(df_test["match_confidence"] > 0.4) & (df_test["match_confidence"] < 0.6)]
    st.dataframe(review_set.sort_values(by="match_confidence").head(10))

    return df_test[df_test["ml_is_match"] == 1]

# === STREAMLIT UI ===
def main():
    st.title("Entity Resolution Demo - Financial Customers")
    st.markdown("""
    **Expected Input File Format (CSV):**
    - Columns: `customer_id`, `full_name`, `last_name`, `address`, `zip`, `email`
    - Both files should have the same structure.
    """)

    df1, df2 = load_data()
    st.write("Data Source 1", df1.head())
    st.write("Data Source 2", df2.head())

    pairs = blocking(df1, df2)
    features, enriched = compute_features(pairs)

    threshold = st.slider("Select rule-based matching threshold", min_value=0, max_value=100, value=80, step=1)
    matched_rule = rule_based_match(enriched.copy(), threshold)
    st.subheader("Rule-Based Matching")
    st.write("Matched Records (Rule-Based)", matched_rule[matched_rule["is_match"] == True])

    st.subheader("ML-Based Matching")
    def label_with_noise(row):
        score = (row["name_sim"] * 0.4 + row["addr_sim"] * 0.4 + row["email_sim"] * 0.2)
        return 1 if score > 80 and np.random.rand() > 0.2 else 0

    enriched["label"] = enriched.apply(label_with_noise, axis=1)
    matched_ml = ml_based_match(enriched.copy(), enriched["label"])

    if not matched_ml.empty:
        st.write("Matched Records (ML-Based)", matched_ml)
    else:
        st.info("No ML-based matches found or insufficient data to train.")

    if st.button("Save Matched Results to CSV"):
        output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/processed'))
        os.makedirs(output_path, exist_ok=True)
        matched_rule[matched_rule["is_match"] == True].to_csv(os.path.join(output_path, "rule_based_matches.csv"), index=False)
        matched_ml.to_csv(os.path.join(output_path, "ml_based_matches.csv"), index=False)
        st.success("Matched results saved to 'data/processed/' folder")

if __name__ == '__main__':
    main()
