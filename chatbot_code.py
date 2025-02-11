from docx import Document
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from datetime import datetime
import tkinter as tk
from tkinter import scrolledtext

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('maxent_ne_chunker')
nltk.download('words')

class ChatBotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("BNCOE ChatBot")
        # Load the Word document
        self.doc = Document("Chatbot.pdf")
        # Extract key-value pairs from the document
        self.key_value_pairs = self.extract_key_value_pairs(self.doc)
        # Read conversation pairs from the text file
        self.common_conversations = self.read_conversation_pairs("conversation_pairs.txt")
        # Clean and preprocess the text data
        self.clean_data = self.preprocess_data()
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer()
        # Fit and transform the preprocessed data
        self.corpus = list(self.clean_data.keys())
        self.X = self.vectorizer.fit_transform(self.corpus)
        # Create chat history display
        self.chat_history = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, state=tk.DISABLED)
        self.chat_history.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        # Create user input field
        self.user_input_field = tk.Text(self.root, height=3, width=50)
        self.user_input_field.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        # Create send button
        self.send_button = tk.Button(self.root, text="Send", width=10, command=self.send_message)
        self.send_button.grid(row=1, column=1, padx=5, pady=5)
        # Bind the Enter key to send_message function
        self.root.bind('<Return>', self.send_message)

        # Configure grid weights for resizing
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Initialize chat history
        self.chat_history.configure(state=tk.NORMAL)
        self.chat_history.tag_config("user", foreground="blue")
        self.chat_history.tag_config("chatbot", foreground="green")
        self.chat_history.insert(tk.END, "ChatBot: Hello! How can I assist you?\n", "chatbot")
        self.chat_history.configure(state=tk.DISABLED)

    def extract_key_value_pairs(self, doc):
        key_value_pairs = {}
        for paragraph in doc.paragraphs:
            if ":" in paragraph.text:
                key, value = paragraph.text.split(":")
                key = key.strip()
                value = value.strip()
                key_value_pairs[key] = value
        return key_value_pairs

    def read_conversation_pairs(self, filename):
        conversation_pairs = {}
        with open(filename, 'r') as file:
            for line in file:
                if ":" in line:
                    key, value = line.strip().split(":")
                    conversation_pairs[key.strip()] = value.strip()
        return conversation_pairs

    def preprocess_text(self, text):
        # Tokenization
        tokens = word_tokenize(text)
        # Convert to lowercase and lemmatize
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens]
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        clean_tokens = [word for word in tokens if word not in stop_words]
        # Reconstruct text from tokens
        clean_text = ' '.join(clean_tokens)
        return clean_text

    def preprocess_data(self):
        cleaned_data = {}
        for question, answer in self.key_value_pairs.items():
            cleaned_question = self.preprocess_text(question)
            cleaned_answer = self.preprocess_text(answer)
            cleaned_data[cleaned_question] = cleaned_answer
        return cleaned_data

    def get_response(self, user_query):
        # Check if the user query matches any common conversational pairs
        for query, response in self.common_conversations.items():
            if query in user_query.lower():
                return response, None
        # Preprocess user query
        cleaned_query = self.preprocess_text(user_query)
        # If no named entities are found, continue with TF-IDF similarity-based response
        query_vector = self.vectorizer.transform([cleaned_query])
        similarities = cosine_similarity(query_vector, self.X)
        most_similar_index = np.argmax(similarities)
        most_similar_question = self.corpus[most_similar_index]
        return self.clean_data.get(most_similar_question, ""), most_similar_question

    def save_response_to_json(self, user_query, response):
        # Define the filename for the JSON file
        json_filename = "chatbot_responses.json"
        # Create a dictionary to represent the user query and response
        data = {
            "timestamp": str(datetime.now()),
            "user_query": user_query,
            "response": response
        }
        # Append the data to the JSON file
        with open(json_filename, 'a') as file:
            json.dump(data, file)
            file.write('\n')

    def send_message(self, event=None):
        user_input = self.user_input_field.get("1.0", tk.END).strip()
        if user_input:
            # Display user message
            self.display_message(user_input, "You", "user")
            # Get chatbot response
            response, _ = self.get_response(user_input)
            # Display chatbot response
            self.display_message(response, "ChatBot", "chatbot")
            # Save user query and chatbot response to JSON
            self.save_response_to_json(user_input, response)
            # Clear user input field
            self.user_input_field.delete("1.0", tk.END)

    def display_message(self, message, sender, tag):
        self.chat_history.configure(state=tk.NORMAL)
        self.chat_history.insert(tk.END, f"{sender}: {message}\n", tag)
        self.chat_history.configure(state=tk.DISABLED)
        self.chat_history.see(tk.END)

# Create main window
root = tk.Tk()
app = ChatBotApp(root)
root.mainloop()
