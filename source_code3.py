# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout

# Step 1: Load the datasets
calls_df = pd.read_csv('calls.csv')  # Replace with the actual path to calls.csv file
reasons_df = pd.read_csv('newReason.csv')  # Replace with the actual path to newReason.csv file

# Step 2: Preprocessing - Select entries with non-empty transcripts
calls_df = calls_df.dropna(subset=['call_transcript'])

# Step 3: Merge datasets based on an identifier (assuming there's a common column to merge on)
df = pd.merge(calls_df, reasons_df, on='id')  

# Step 4: Prepare the text data (call transcripts) and labels (primary call reasons)
X = df['call_transcript'].values  
y = df['primary_call_reason'].values  

# Step 5: Tokenization and Padding for X (transcripts)
max_words = 5000  
max_len = 200 

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)
X_padded = pad_sequences(X_seq, maxlen=max_len)

# Step 6: Encoding labels y (primary call reasons) using LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)  
y_onehot = to_categorical(y_encoded)  

# Step 7: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_padded, y_onehot, test_size=0.2, random_state=42)

# Step 8: Building the LSTM model
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=128, input_length=max_len))
model.add(LSTM(64, return_sequences=False))   
model.add(Dropout(0.5))  
model.add(Dense(len(le.classes_), activation='softmax'))  

# Step 9:Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 10:Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)

# Step 11:Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Step 12: Make predictions on new data

predictions = model.predict(X_test)

# Convert the predicted probabilities back to the original class labels
predicted_classes = np.argmax(predictions, axis=1)
predicted_labels = le.inverse_transform(predicted_classes)

# Print out some example predictions
for i in range(5):
    print(f"Predicted Reason: {predicted_labels[i]}")
    print(f"Original Transcript: {X[i]}")
    print()
