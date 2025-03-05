import tkinter as tk
from tkinter import ttk, scrolledtext
import joblib
import os
import re

# Зареждане на модела и векторизатора
# Уверете се, че файловете съществуват в посочените пътища
model_path = './models/best_sentiment_model.pkl'  # Моделът за предсказване на настроения
vectorizer_path = './models/vectorizer.pkl'  # Векторизатор за обработка на текст
model = joblib.load(model_path)  # Зареждане на модела
vectorizer = joblib.load(vectorizer_path)  # Зареждане на векторизатора

# Дефиниция на функцията за почистване на текст
# Премахване на линкове, хаштагове и специални символи
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Премахване на URL адреси
    text = re.sub(r'\@\w+|\#', '', text)  # Премахване на хаштагове и @
    text = re.sub(r'[^\w\s]', '', text)  # Премахване на специални символи
    text = re.sub(r'\s+', ' ', text)  # Премахване на излишни интервали
    return text.strip().lower()  # Връщане на изчистен текст в малки букви

# Функция за предсказване на настроението
def predict_sentiment():
    text = input_textbox.get("1.0", tk.END).strip()  # Вземане на въведения текст
    if text:  # Проверка дали е въведен текст
        cleaned_text = clean_text(text)  # Почистване на текста
        vect_text = vectorizer.transform([cleaned_text])  # Векторизиране на текста
        prediction = model.predict(vect_text)  # Предсказване на настроението
        sentiment = {1: 'Положително', 0: 'Неутрално', -1: 'Отрицателно'}  # Карта на настроенията
        result_label.config(text=f"Настроение: {sentiment[prediction[0]]}")  # Показване на резултата
    else:
        result_label.config(text="Моля, въведете текст.")  # Съобщение за липсващ текст

# Създаване на главния прозорец
root = tk.Tk()
root.title("Класификация на настроенията")  # Заглавие на прозореца
root.resizable(False, False)  # Забраняване на разширяване и смаляване

# Стайлинг на интерфейса
style = ttk.Style()
style.configure('TFrame', background='#f0f0f0')  # Цвят на фона на рамката
style.configure('TButton', font=('Arial', 10), background='#e1e1e1', width=20)  # Стил на бутона
style.configure('TLabel', background='#f0f0f0', font=('Arial', 12))  # Стил на етикета
style.configure('TScrolledText', font=('Consolas', 10))  # Стил на текстовото поле

# Създаване на рамка за въвеждане на текст
frame = ttk.Frame(root, padding=20)
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
frame.columnconfigure(0, weight=1)  # Указване, че колоната се разтяга равномерно

# Текстово поле за въвеждане на текст
input_textbox = scrolledtext.ScrolledText(frame, width=60, height=10, font=('Consolas', 12))
input_textbox.grid(row=0, column=0, columnspan=2, pady=10, padx=5)

# Бутон за предсказване на настроението
predict_button = ttk.Button(frame, text="Анализирай", command=predict_sentiment)
predict_button.grid(row=1, column=0, pady=10, sticky=tk.N)  # Центриране на бутона над етикета

# Етикет за показване на резултата
result_label = ttk.Label(frame, text="Настроение: Неизвестно")
result_label.grid(row=2, column=0, pady=10, sticky=tk.N)  # Подравняване на етикета под бутона

# Стартиране на приложението
root.mainloop()


#python sentiment_analyzer_app.py