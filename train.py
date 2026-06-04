import pandas as pd
import numpy as np
import pickle
import re
import nltk

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('stopwords')

# Load dataset
df = pd.read_csv('C:/Users/asus/OneDrive/Documents/Sems 6/TEXT MINING/ASG_Sesi12/data.csv')

# ====== SESUAIKAN KOLOM DI SINI ======
text_col = 'text'
label_col = 'label'
# =====================================

# Preprocessing
stop_words = set(stopwords.words('indonesian')) - {"tidak", "bukan"}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df[text_col] = df[text_col].apply(clean_text)

# Tokenizer
max_words = 10000
max_len = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df[text_col])

X = tokenizer.texts_to_sequences(df[text_col])
X = pad_sequences(X, maxlen=max_len)

y = df[label_col].values

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model LSTM
from tensorflow.keras.layers import Bidirectional

model = Sequential()
model.add(Embedding(max_words, 128))
model.add(Bidirectional(LSTM(128)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=8, batch_size=32, validation_data=(X_test, y_test))

# Save model
model.save("model.h5")

# Save tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Training selesai!")