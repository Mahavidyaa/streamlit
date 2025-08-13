import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.title("📊 Simple Linear Regression - Exam Score Prediction")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv", "txt"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("📂 Dataset Preview")
    st.write(data.head())

    if "Hours_Studied" in data.columns and "Exam_Score" in data.columns:
        # Prepare data
        X = data[['Hours_Studied']]
        y = data['Exam_Score']

        # Train-test split
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        model = LinearRegression()
        model.fit(x_train, y_train)

        # Show training/testing data
        st.subheader("📋 Training Data")
        st.write(pd.concat([x_train, y_train], axis=1).head())

        st.subheader("📋 Testing Data")
        st.write(pd.concat([x_test, y_test], axis=1).head())

        # Predictions
        y_pred = model.predict(x_test)

        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(x_test, y_test, color='green', label='Actual Scores')
        ax.plot(x_test, y_pred, color='red', label='Regression Line')
        ax.set_xlabel('Hours Studied')
        ax.set_ylabel('Exam Score')
        ax.set_title('Actual vs Predicted Scores')
        ax.legend()
        st.pyplot(fig)

        # User input for prediction
        st.subheader("🎯 Predict Exam Score")
        hours = st.number_input("Enter hours studied:", min_value=0.0, step=0.5)
        if st.button("Predict"):
            pred_score = model.predict([[hours]])
            st.success(f"Predicted Exam Score for {hours} hours studied: {pred_score[0]:.2f}")
    else:
        st.error("The dataset must contain 'Hours_Studied' and 'Exam_Score' columns.")
else:
    st.info("Please upload a CSV file to continue.")
