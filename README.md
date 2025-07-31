# 🧠 Medical AI Chatbot

A voice-based AI chatbot that takes medical symptom input via voice, optionally accepts image uploads (e.g., skin rash), and provides curative and preventive health advice using AI and rule-based logic.

---

🚀 Features

- 🎤 Voice Input (using Speech Recognition)
- 🧾 Text Input
- 🖼️ Image Upload (for optional skin disease diagnosis)
- 🩺 Symptom Checker using Machine Learning
- 🔊 Voice Output using gTTS or pyttsx3
- 📋 Textual medical advice (curative & preventive)

---

 🛠️ Tech Stack

- Frontend: Gradio
- Backend: Python, scikit-learn, pandas
- Speech: SpeechRecognition, pyttsx3 / gTTS
- Image Support: OpenCV / CNN (optional)
- Model: Trained ML model (RandomForestClassifier or SVM)
- Deployment: Localhost or Google Colab

---

⚙️ Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/medicalAI_chatbot.git
   cd medicalAI_chatbot
2. Create virtual environment (optional but recommended):
   python -m venv venv
   venv\Scripts\activate  # For Windows
3. Install dependencies:
   pip install -r requirements.txt
4. python app.py


📁 File Structure

medicalAI_chatbot/
│
├── app.py                     # Main Gradio interface
├── train_model.py             # Model training script
├── chatbot_utils.py           # Custom logic for prediction
├── requirements.txt           # Python packages
├── model.pkl                  # Trained model
├── README.md                  # Project info
└── assets/                    # Optional: images or supporting files


💡 Future Improvements
-Integrate BERT for better symptom understanding
-Add disease severity prediction
-Support for multilingual inputs
-Online deployment on Hugging Face or Streamlit Cloud


👩‍⚕️ Disclaimer
This chatbot is for educational purposes only. It does not replace professional medical advice.

.

📝 License
This project is open-source under the MIT License.

