import os
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

# Дефиниране на пътя за запазване на моделите
save_directory = './models'
os.makedirs(save_directory, exist_ok=True)

# Зареждане на данни
train_file = 'train.csv'
test_file = 'test.csv'

train_data = pd.read_csv(train_file, encoding='latin1')
test_data = pd.read_csv(test_file, encoding='latin1')

# Прочистване на данни
train_data.dropna(subset=['text', 'sentiment'], inplace=True)
test_data.dropna(subset=['text', 'sentiment'], inplace=True)

train_data['sentiment'] = train_data['sentiment'].map({'positive': 1, 'negative': -1, 'neutral': 0})
test_data['sentiment'] = test_data['sentiment'].map({'positive': 1, 'negative': -1, 'neutral': 0})

# Почистване на текст

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Премахване на линкове
    text = re.sub(r'\@\w+|\#', '', text)  # Премахване на хаштагове и споменавания
    text = re.sub(r'[^\w\s]', '', text)  # Премахване на пунктуация
    text = re.sub(r'\s+', ' ', text)  # Премахване на допълнителни интервали
    return text.strip().lower()

train_data['text'] = train_data['text'].apply(clean_text)
test_data['text'] = test_data['text'].apply(clean_text)

# Разделяне на данните
X_train = train_data['text']
y_train = train_data['sentiment']
X_test = test_data['text']
y_test = test_data['sentiment']

# Намаляване на обема на тренировъчните данни
X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=0.5, random_state=42)

# Векторизация
vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words='english', sublinear_tf=True)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Инициализация на модели
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "SVM": SVC(kernel='linear', class_weight='balanced', probability=True),
    "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
}

# Резултати от модели
results = {}
best_model = None
best_accuracy = 0

for name, model in models.items():
    print(f"\nОбучение и тестване на модел: {name}")
    model.fit(X_train_vect, y_train)
    predictions = model.predict(X_test_vect)
    accuracy = accuracy_score(y_test, predictions)
    results[name] = accuracy

    print(f"\n{name} Classification Report:\n")
    print(classification_report(y_test, predictions, zero_division=0))

    # Визуализация на матрицата на объркване
    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, display_labels=['Negative', 'Neutral', 'Positive']
    )
    plt.title(f"Confusion Matrix: {name}")  # Заглавие на графиката
    plt.gcf().canvas.manager.set_window_title(f"Confusion Matrix: {name}")  # Заглавие на прозореца
    plt.show()

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

# Резултати от всички модели
print("\nРезултати от всички модели:")
for model_name, acc in results.items():
    print(f"{model_name}: {acc:.4f}")

# Запазване на най-добрия модел и векторизатора
print("\nЗапазване на най-добрия модел...")
model_path = os.path.join(save_directory, 'best_sentiment_model.pkl')
vectorizer_path = os.path.join(save_directory, 'vectorizer.pkl')

joblib.dump(best_model, model_path)
joblib.dump(vectorizer, vectorizer_path)

print(f"Най-добрият модел е запазен като '{model_path}'")
print(f"Векторизаторът е запазен като '{vectorizer_path}'")

# Тестване на най-добрия модел с нови данни
print("\nТестване на най-добрия модел с нови текстове...")
loaded_model = joblib.load(model_path)
loaded_vectorizer = joblib.load(vectorizer_path)

new_texts = [
    "I am sick.",
    "I am sick but I am happy.",
    "I love this!",
    "This is terrible...",
    "The sky is blue.",
    "This product is amazing!",
    "I hate this so much.",
    "It's neither good nor bad.",
    "I am sad.",
    "Тhis is the best experience ever!",
    "I'm so happy with this product.",
    "Amazing quality and service!",
    "I'm in love with this design.",
    "Great job, this is awesome!",
    "This is a masterpiece!This is the worst thing I've ever seen.",
    "I absolutely hate this!",
    "Terrible experience, very disappointed.",
    "Awful service, I will never return.",
    "This is utterly horrible.",
    "I'm so frustrated with this.",
    "I have no strong feelings about this.",
    "It works as expected, no complaints.",
    "It's neither good nor bad.",
    "This is just average.",
    "This is acceptable, nothing more.",
    "It's alright, nothing remarkable.",
    "2am feedings for the baby are fun when he is all smiles and coos. It’s moments like these that make the sleepless nights worthwhile. However, I’m worried about running out of baby supplies soon. Time to plan another shopping trip.",
    "Journey!? Wow... you just became cooler. Hehe... (is that possible!?) This day has been amazing so far. I even discovered a new app on my iPod, and I’m completely hooked on it. Highly recommend giving it a try!",
    "What interview! Leave me alone. I can’t handle this pressure right now. On the bright side, at least the weather outside is lovely. Maybe a walk will help clear my mind and get me focused again.",
    "I’d have responded, if I were going. But plans changed, and now I’m stuck at home. Not the most exciting day, but at least I got time to clean up my desk. Small wins, I guess!",
    "Sooo SAD I will miss you here in San Diego!!! I can’t believe we didn’t have more time together. It feels so unfair that goodbyes always come too soon.",
    "What interview! Leave me alone. The stress is unbearable, and I feel like I’m being pushed to my limit. I wish I could just disappear for a while and escape all of this.",
    "My Sharpie is running DANGERously low on ink. Just my luck! Every time I need something to work, it fails on me. Can’t anything go right for once?",
    "Uh oh, I am sunburned. I knew I should have used sunscreen, but I didn’t. Now my skin is burning, and I feel so uncomfortable. This day is officially ruined.",
    "I really love how today turned out! The weather was perfect, and I finally had a chance to meet up with old friends. We laughed so much, and I feel so grateful for moments like these.",
    "The free fillin’ app on my iPod is so fun, I can’t stop playing it. I haven’t been this hooked on something in a long time. Highly recommend giving it a try if you need some entertainment!",
    "A little happy for the wine jeje. It’s my free time, so who cares? This day has been absolutely amazing, and I’m ending it with a big smile on my face.",
    "I’d have responded, if I were going. Plans just didn’t work out this time. It’s okay, though; maybe next time something better will come up. For now, I’ll just stay in and relax.",
    "Test test from the LG enV2. Just trying out this new gadget to see how it works. Nothing extraordinary so far, but it gets the job done. Let’s see if it grows on me."
    
]

new_texts_cleaned = [clean_text(text) for text in new_texts]
new_texts_vect = loaded_vectorizer.transform(new_texts_cleaned)
new_predictions = loaded_model.predict(new_texts_vect)

print("\nРезултати за нови текстове:")
print("-1 е за отрицателен, 0 неутрален, 1 положителен\n")

for text, sentiment in zip(new_texts, new_predictions):
    print(f"Текст: '{text}' -> Настроение: {sentiment}")



#python sentiment_model_training.py