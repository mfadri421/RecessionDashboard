# Save this content as app.py

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# data_processing.py functions (included for persistence)
def load_data(filepath):
    """Loads data from a CSV file."""
    df = pd.read_csv(filepath, parse_dates=["Date"])
    return df

def preprocess_data(df, scaler=None, is_training=True, seq_len=6):
    """
    Preprocesses the data by calculating derivatives, scaling, and creating sequences.

    Args:
        df (pd.DataFrame): The input DataFrame.
        scaler (StandardScaler, optional): Fitted scaler object. Defaults to None.
        is_training (bool, optional): Whether the data is for training (fits scaler). Defaults to True.
        seq_len (int, optional): The sequence length for time series data. Defaults to 6.

    Returns:
        tuple: A tuple containing:
            - tf.Tensor: The preprocessed and sequenced features.
            - tf.Tensor: The labels (if is_training is True), otherwise None.
            - StandardScaler: The fitted scaler object (if is_training is True), otherwise the provided scaler.
            - np.ndarray: The original dates corresponding to the sequenced data.
    """
    X = df.drop(columns=["Date", "RecessionLabel"])
    y = df["RecessionLabel"]

    # Calculate first derivatives
    X_diff = X.diff().fillna(0)
    X_diff.columns = [f"{col}_diff" for col in X_diff.columns]
    X_combined = pd.concat([X, X_diff], axis=1)

    # Scale the data
    if is_training:
        scaler = StandardScaler()
        X_scaled_np = scaler.fit_transform(X_combined)
    else:
        X_scaled_np = scaler.transform(X_combined)

    # Convert to TensorFlow tensor
    X_scaled = tf.convert_to_tensor(X_scaled_np, dtype=tf.float32)
    y_tensor = tf.convert_to_tensor(y.values, dtype=tf.float32)

    # Reshape for sequence input: [samples, timesteps, features]
    X_seq = tf.stack([X_scaled[i:i+seq_len] for i in range(len(X_scaled) - seq_len + 1)])
    y_seq = y_tensor[seq_len-1:] # Adjust label slicing for end of sequence

    # Get dates for the sequenced data
    dates_seq = df["Date"].values[seq_len-1:]


    if is_training:
        return X_seq, y_seq, scaler, dates_seq
    else:
        return X_seq, None, scaler, dates_seq


def build_transformer_model(seq_len, n_features):
    """Builds the transformer model."""
    inputs = layers.Input(shape=(seq_len, n_features))
    x = layers.LayerNormalization()(inputs)
    x = layers.MultiHeadAttention(num_heads=2, key_dim=4)(x, x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, outputs)
    return model

def train_model(model, X_train, y_train, epochs=50, batch_size=16, validation_split=0.1):
    """Trains the transformer model."""
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=0)
    return model, history

def predict_recession(model, X_seq):
    """Makes recession probability predictions."""
    y_pred_prob = model.predict(X_seq).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)
    return y_pred_prob, y_pred

def plot_rolling_forecast(dates, predicted_probabilities, true_labels=None, title="Rolling Recession Probability Forecast"):
    """Plots the rolling forecast and optionally true labels."""
    plt.figure(figsize=(12, 6))
    plt.plot(dates, predicted_probabilities, label="Predicted Probability")
    if true_labels is not None:
        plt.plot(dates, true_labels, label="True Recession", linestyle='--')
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Probability")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig = plt.gcf()
    plt.close()
    return fig

def plot_confusion_matrix(true_labels, predicted_labels, title="Confusion Matrix"):
    """Plots the confusion matrix."""
    cm = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Recession", "Recession"])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax)
    plt.title(title)
    fig = plt.gcf()
    plt.close()
    return fig


# Set Matplotlib backend (although Streamlit handles this, explicit setting might be needed in some environments)
plt.switch_backend('Agg')

# Set the sequence length
SEQ_LEN = 6

# ...existing imports...

# Streamlit file uploader for new data
uploaded_file = st.file_uploader("Upload new recession data CSV", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.success("New data loaded!")
else:
    df = load_data("final_recession_dataset_1975_2025.csv")
# ...rest of your code...

# Preprocess data and train the model
X_seq, y_seq, scaler, dates_seq = preprocess_data(df, seq_len=SEQ_LEN)
n_features = X_seq.shape[-1]
model = build_transformer_model(SEQ_LEN, n_features)
model, history = train_model(model, X_seq, y_seq)

# Generate rolling forecasts
# We need to generate rolling forecasts for the entire period where a 6-month sequence can be formed
forecast_dates = []
predicted_probabilities = []

# Loop through the DataFrame starting from the earliest possible forecast index
# up to the end, excluding the last SEQ_LEN rows since we need SEQ_LEN rows to make a prediction
earliest_forecast_index = SEQ_LEN - 1 # Adjusted index to align with the dates_seq
for i in range(earliest_forecast_index, len(df)):
    # Select the 6 months of data preceding the current date (index i+1), including index i
    current_sequence_df = df.iloc[i - SEQ_LEN + 1 : i + 1].copy()

    # Only proceed if we have a full sequence of length SEQ_LEN
    if len(current_sequence_df) == SEQ_LEN:
        # Preprocess the current sequence using the fitted scaler
        current_X_seq, _, _, _ = preprocess_data(current_sequence_df, scaler=scaler, is_training=False, seq_len=SEQ_LEN)

        # Predict recession probability
        current_pred_prob = model.predict(current_X_seq).flatten()

        # Store the current date and the predicted probability
        forecast_dates.append(df["Date"].iloc[i])
        predicted_probabilities.append(current_pred_prob[0])


rolling_forecast_df = pd.DataFrame({
    "Date": forecast_dates,
    "Predicted_Probability": predicted_probabilities
})


# Dashboard Layout
st.title("Recession Probability Forecast Dashboard")

# Add a date range slider for filtering the rolling forecast plot
min_date = rolling_forecast_df["Date"].min().to_pydatetime()
max_date = rolling_forecast_df["Date"].max().to_pydatetime()

date_range = st.slider(
    "Select Date Range for Rolling Forecast Plot:",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    step=pd.Timedelta(days=30),
    format="YYYY-MM"
)

# Filter the rolling forecast DataFrame based on the selected date range
filtered_rolling_forecast_df = rolling_forecast_df[
    (rolling_forecast_df["Date"] >= date_range[0]) & (rolling_forecast_df["Date"] <= date_range[1])
].copy()

# Align the true labels with the filtered forecast dates
# Find the starting index in the original df that corresponds to the start date of the filtered data
start_index_filtered = df[df["Date"] == filtered_rolling_forecast_df["Date"].iloc[0]].index[0]
true_labels_aligned_filtered = df["RecessionLabel"].values[start_index_filtered : start_index_filtered + len(filtered_rolling_forecast_df)]


# Display Rolling Forecast Plot
st.header("Rolling 6-Month Recession Probability Forecast")
rolling_forecast_fig = plot_rolling_forecast(
    filtered_rolling_forecast_df["Date"],
    filtered_rolling_forecast_df["Predicted_Probability"],
    true_labels=true_labels_aligned_filtered
)
st.pyplot(rolling_forecast_fig)

# Evaluate and Display Confusion Matrices
st.header("Model Evaluation")

# Predictions and evaluation on the entire dataset
y_pred_prob_all, y_pred_all = predict_recession(model, X_seq)
cm_all_fig = plot_confusion_matrix(y_seq.numpy(), y_pred_all, title="Confusion Matrix - All Data")
st.pyplot(cm_all_fig)

# For the test set, we need to re-split the data to get the original test set indices or data
# As the previous code didn't return the train/test split, we'll re-perform the split for evaluation purposes.
# This is not ideal for a production dashboard but serves the purpose of displaying test set performance.
split_idx = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

y_pred_prob_test, y_pred_test = predict_recession(model, X_test)
cm_test_fig = plot_confusion_matrix(y_test.numpy(), y_pred_test, title="Confusion Matrix - Test Set")
st.pyplot(cm_test_fig)
