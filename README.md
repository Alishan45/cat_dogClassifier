# 🐾 Cat vs Dog Classifier (CNN & Hybrid CNN+MLP)  

This project is a **deep learning web app** built with **Streamlit** and **TensorFlow** that classifies images of **cats vs dogs** using two different models:  
1. A **Convolutional Neural Network (CNN)**  
2. A **Hybrid model (CNN + MLP)**  

The app allows users to **upload images** and see predictions from both models side by side.  

---

## 🚀 Features  
- Train and evaluate CNN and Hybrid (CNN+MLP) models  
- Visualize **training vs validation accuracy/loss curves**  
- Display **confusion matrix** after training  
- Upload images and get predictions with probability scores  
- Interactive **Streamlit UI**  

---

## 🛠️ Tech Stack  
- [TensorFlow](https://www.tensorflow.org/)  
- [Keras](https://keras.io/)  
- [Streamlit](https://streamlit.io/)  
- [NumPy](https://numpy.org/)  
- [Pandas](https://pandas.pydata.org/)  
- [Matplotlib](https://matplotlib.org/)  
- [Scikit-learn](https://scikit-learn.org/stable/)  

---

## 📂 Project Structure

├── app.py                # Main Streamlit app
├── models/
│   ├── cnn\_model.h5      # Saved CNN model
│   ├── hybrid\_model.h5   # Saved Hybrid CNN+MLP model
├── dataset/              # Cat & Dog dataset (not uploaded here)
├── requirements.txt      # Dependencies
├── README.md             # Project documentation

````

---


2. **Create a virtual environment (Python 3.10/3.11 recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Usage

1. **Train the models** (CNN & Hybrid)

   ```bash
   python train.py
   ```

2. **Run the Streamlit app**

   ```bash
   streamlit run app.py
   ```

3. Open your browser at **`http://localhost:8501`**

---

## 📊 Results

* **CNN Model**: 82% accuracy on test set
* **Hybrid CNN+MLP Model**: 80% accuracy on test set
* Confusion Matrix and Accuracy/Loss curves available in app

---
## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to improve.

---

## 📜 License

[MIT](LICENSE)

---

## 👨‍💻 Author

* Ali Shan – [GitHub](https://github.com/Alishan45) | [LinkedIn](https://www.linkedin.com/in/ali-shan-542246235/)

```

---

