# bank_customer_churn_prediction
A Streamlit-based web app for predicting customer churn using an Artificial Neural Network (ANN). Built with TensorFlow, Scikit-learn, and pre-trained models and encoders for real-time predictions.
# 🧠 Customer Churn Prediction App (Streamlit + ANN)

This is a simple and interactive web application built with **Streamlit** that predicts whether a customer is likely to churn from a bank based on their profile data. The model behind the scenes is a trained **Artificial Neural Network (ANN)** using TensorFlow/Keras.

---

## 🚀 Features

- Predict churn likelihood using user-friendly inputs
- Uses a trained ANN (`model.h5`) for prediction
- Preprocessing with Scikit-learn:
  - LabelEncoder for gender
  - OneHotEncoder for geography
  - StandardScaler for input normalization
- Real-time probability score of churn
- Clean UI powered by Streamlit

---

## 🧰 Tech Stack

- **Frontend**: Streamlit
- **Model**: TensorFlow / Keras ANN
- **Preprocessing**: Scikit-learn
- **Language**: Python

---

## 📂 Folder Structure

annclassification/ ├── app.py # Streamlit app ├── model.h5 # Trained ANN model ├── scaler.pkl # StandardScaler for numeric features ├── label_encoder_gender.pkl # LabelEncoder for gender ├── onehot_encoder_geo.pkl # OneHotEncoder for geography ├── requirements.txt # Python dependencies └── README.md # Project overview

## ▶️ Running the App Locally

```bash
pip install -r requirements.txt
streamlit run app.py
**** Visit: http://localhost:8501

🌐 Deployment

You can deploy this app to:

    Streamlit Cloud

    A Virtual Machine (e.g., Alibaba Cloud ECS)

    Docker container

    Azure App Service / Heroku / AWS EC2

📊 Sample Prediction Output

    Input features: Credit Score, Age, Gender, Geography, etc.

    Output: Churn Probability: 0.78

    Message: The customer is likely to churn.

🔒 Note

Ensure the .pkl and .h5 model files are present in the same directory when running the app. These files are critical for data transformation and prediction.

📬 Contact

Feel free to fork or contribute. For issues, please open an issue or pull request!
