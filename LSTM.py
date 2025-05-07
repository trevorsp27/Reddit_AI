import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

#load and clean data
df = pd.read_csv("reddit_posts_labeled_clean.csv")

def parse_reliable(val):
    if isinstance(val, str):
        val = val.strip()
        if "True" in val:
            return 1
        elif "False" in val:
            return 0
    return int(val)

df['reliable'] = df['reliable'].apply(parse_reliable)
df = df.dropna(subset=['title', 'body', 'reliable'])
df['text'] = df['title'].astype(str) + " " + df['body'].astype(str)

texts = df['text'].tolist()
labels = df['reliable'].astype(int).tolist()

#tokenization
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, padding='post', maxlen=80)

#train test split
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, labels, test_size=0.2, stratify=labels, random_state=42
)
y_train = np.array(y_train)
y_test = np.array(y_test)

#  weights
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

#model
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=64, input_length=80))
model.add(Bidirectional(LSTM(64, return_sequences=False, kernel_regularizer=l2(0.001))))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.001)))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

#train
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=10,
    batch_size=32,
    class_weight=class_weights_dict,
    callbacks=[early_stopping]
)

#evaluate
y_pred = (model.predict(X_test) > 0.5).astype(int)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

#curve
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve Over Epochs')
plt.legend()
plt.show()
