Breast Cancer Detection Web App

An AI-powered web application that predicts whether a tumor is benign or malignant using machine learning, helping in early detection and decision support.

Features
1. Predicts breast cancer (Benign / Malignant) using ML model
2. User-friendly interface for inputting medical parameters
3. Real-time prediction with high accuracy (~95%+)
4. Data preprocessing using NumPy & Pandas
5. REST API integration with FastAPI
6. Tech Stack
   
Frontend: HTML, CSS, JavaScript
Backend: FastAPI
ML Model: Scikit-learn
Data Handling: NumPy, Pandas

Project Structure
breast-cancer-detector/
│── frontend/
│── backend/
│   ├── main.py
│   ├── model.pkl
│   └── utils.py
│── requirements.txt
│── README.md
 How It Works
User enters tumor-related features
Data is preprocessed using Pandas/NumPy
ML model predicts the result
Output displayed as Benign or Malignant
Run Locally
# Install dependencies
pip install -r requirements.txt

# Run FastAPI server
uvicorn main:app --reload

Open: http://127.0.0.1:8000/docs

Model Details
Algorithm: Logistic Regression / Random Forest
Dataset: Breast Cancer Wisconsin Dataset
Accuracy: ~95% (can vary based on training)

Future Improvements
Add visualization dashboard 
Integrate explainable AI (SHAP/LIME)
Deploy on cloud (AWS/GCP)
Add user authentication
