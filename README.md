## **N-Gram Language Model with Streamlit Deployment**

Proyek ini adalah implementasi **model bahasa berbasis N-Gram** menggunakan pustaka Python seperti `nltk`. Model ini mampu menganalisis teks, menghasilkan N-Gram (1-gram hingga 5-gram), dan melatih model bahasa untuk memprediksi kata berikutnya berdasarkan konteks.

### **Fitur Utama**
1. **Preprocessing Teks**: Membersihkan teks dari simbol, angka, dan URL untuk memastikan kualitas data input.
2. **Pembuatan N-Gram**: Menghasilkan daftar N-Gram (dengan frekuensi) dari teks yang telah dibersihkan.
3. **Model Bahasa**: Melatih model bahasa berbasis Maximum Likelihood Estimation (MLE) untuk memprediksi kata berikutnya.
4. **Generasi Kalimat**: Menghasilkan kalimat acak menggunakan kata-kata yang diprediksi dari model.
5. **Deploy dengan Streamlit**: Aplikasi dilengkapi antarmuka pengguna untuk menjalankan model secara interaktif melalui web.

### **Alur Kerja**
1. Unggah file teks sebagai input.
2. Pilih nilai `n` untuk menghasilkan N-Gram (1-gram hingga 5-gram).
3. Model akan:
   - Membersihkan teks.
   - Melatih model bahasa berbasis N-Gram.
   - Menyimpan N-Gram dan frekuensi ke file.
   - Menghasilkan kalimat acak berdasarkan model bahasa.
4. Hasilnya akan ditampilkan di antarmuka Streamlit.

### **Tujuan Proyek**
Proyek ini dirancang untuk memahami cara kerja model bahasa berbasis N-Gram dan aplikasinya dalam pemrosesan bahasa alami (NLP). Dengan menggunakan antarmuka Streamlit, pengguna dapat dengan mudah bereksperimen dan mengevaluasi hasil dari model yang dibangun.
