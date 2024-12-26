import streamlit as st
import nltk
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.probability import FreqDist
import pickle
import re
import random
from nltk.tokenize.treebank import TreebankWordDetokenizer

# Pastikan resource NLTK sudah diunduh
nltk.download('punkt')

# Detokenizer
detokenize = TreebankWordDetokenizer().detokenize

# Fungsi untuk membersihkan teks
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Melatih model bahasa
def train_language_model(text, n):
    tokenized_text = [word_tokenize(sent) for sent in nltk.sent_tokenize(text)]
    train_data, vocab = padded_everygram_pipeline(n, tokenized_text)
    model = MLE(n)
    model.fit(train_data, vocab)
    return model

# Menyimpan n-gram dan frekuensi
def save_ngrams_freq(cleaned_text, n):
    tokens = nltk.word_tokenize(cleaned_text)
    ngram_freq = FreqDist(ngrams(tokens, n))
    # Konversi kunci tuple menjadi string
    formatted_freq = {str(ngram): freq for ngram, freq in ngram_freq.items()}
    return formatted_freq

# Menghasilkan kalimat dengan kata awal acak
def generate_sent(model, num_words):
    content = []
    for token in model.generate(num_words, random_seed=random.randint(1, 100)):
        if token == '<s>':
            continue
        if token == '</s>':
            break
        content.append(token)
    return detokenize(content)

# Aplikasi Streamlit
st.title("N-grams Language Model with MLE")

# Upload file teks
uploaded_file = st.file_uploader("Unggah file teks Anda", type=["txt"])
if uploaded_file:
    # Membaca file
    text = uploaded_file.read().decode("utf-8")
    
    # Pembersihan teks
    cleaned_text = clean_text(text)
    if st.checkbox("Tampilkan teks yang telah dibersihkan"):
        st.write(cleaned_text)

    # Input nilai n
    n = st.number_input("Masukkan nilai n untuk n-grams (1-5):", min_value=1, max_value=5, step=1)
    
    if st.button("Generate Model dan Kalimat"):
        # Melatih model bahasa
        language_model = train_language_model(cleaned_text, n)
        st.success(f"Model {n}-grams berhasil dibuat!")

        # Menyimpan frekuensi n-grams
        ngram_freq = save_ngrams_freq(cleaned_text, n)
        st.write(f"Frekuensi {n}-grams (contoh 10 teratas):")
        st.json(dict(list(ngram_freq.items())[:10]))  # Gunakan st.json untuk menampilkan data JSON-friendly

        # Menghasilkan kalimat
        generated_sentence = generate_sent(language_model, 10)
        st.write("### Kalimat yang dihasilkan:")
        st.write(generated_sentence)

else:
    st.info("Silakan unggah file teks untuk memulai.")
