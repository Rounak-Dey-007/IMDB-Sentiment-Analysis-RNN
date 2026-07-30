# IMDB Sentiment Analysis using RNN
A PyTorch-based Recurrent Neural Network (RNN) that classifies IMDB movie reviews as **positive** or **negative**. The pipeline covers text cleaning, TF-IDF vectorization, a custom RNN architecture, training, and evaluation — achieving **87.58% test accuracy**.

## 📌 Overview
This project builds a sentiment classifier from scratch using a many-to-one RNN architecture trained on TF-IDF-vectorized movie reviews. It walks through the full NLP workflow: raw text → cleaned text → numerical features → trained model → evaluated predictions.

## 📂 Dataset
- **Source:** [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) (Kaggle)
- **Size:** 50,000 reviews, labeled `positive` / `negative`
- **Columns:** `review` (text), `sentiment` (label)

> The dataset CSV is not included in this repo due to size. Download it from Kaggle and place it as `data/IMDB Dataset.csv` (see [Setup](#-setup--usage) below).

## ⚙️ Workflow
1. **Load the dataset** — read the CSV into a Pandas DataFrame
2. **Exploratory analysis** — check for nulls, HTML tags, and URLs in review text
3. **Text preprocessing**
   - Lowercasing
   - URL removal
   - HTML tag & punctuation removal
   - Stopword removal (NLTK)
   - Stemming (Porter Stemmer)
4. **Label encoding** — `positive`/`negative` → `1`/`0` via `LabelEncoder`
5. **Feature extraction** — `TfidfVectorizer` (`max_features=10000`)
6. **Train/test split** — 80/20 split via `train_test_split`
7. **Data loading** — wrapped into PyTorch `TensorDataset` + `DataLoader` (batch size 64)
8. **Model architecture** — custom RNN (see below)
9. **Training** — 15 epochs, Adam optimizer, BCE loss
10. **Evaluation** — accuracy computed on held-out test set

- **Type:** Single-layer vanilla RNN, many-to-one
- **Hidden size:** 128
- **Optimizer:** Adam (`lr=0.0001`)
- **Loss:** Binary Cross-Entropy (`BCELoss`, applied after `sigmoid`)
- **Epochs:** 15

## 📊 Results

| Metric | Value |
|---|---|
| Test Accuracy | **87.58%** |

Training loss decreased steadily across epochs (see the loss curve plotted in the notebook).

## 🔮 Future Improvements
- Replace TF-IDF with pretrained embeddings (GloVe / Word2Vec) or a learned `nn.Embedding` layer
- Try LSTM / GRU / Bidirectional RNN to better capture long-range dependencies
- Add validation split and early stopping
- Hyperparameter tuning (hidden size, learning rate, number of layers)
- Save/load the trained model (`torch.save` / `torch.load`) for inference on new reviews
