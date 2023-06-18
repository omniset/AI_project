import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Baca dataset IMDb
df = pd.read_csv('imdb_top_1000.csv', delimiter=';')

# Preprocessing data
df['genres'] = df['genres'].str.split('|').apply(lambda x: ' '.join(x))

# Menggunakan Bag-of-Words untuk mengubah genre film menjadi vektor fitur
vectorizer = CountVectorizer()
genre_matrix = vectorizer.fit_transform(df['genres'])

# Menghitung kemiripan kosinus antara genre film
cosine_sim = cosine_similarity(genre_matrix)

def get_recommendations(title, cosine_sim, df, top_n=10):
    # Mendapatkan indeks film berdasarkan judul
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()

    # Mendapatkan indeks film yang sesuai dengan judul input
    idx = indices[title]

    # Mendapatkan similarity scores dari film-film yang lain berdasarkan genre
    genre_scores = list(enumerate(cosine_sim[idx]))

    # Mendapatkan similarity scores dari film-film yang lain berdasarkan IMDb rating
    imdb_scores = list(enumerate(df['IMDB_Rating']))

    # Mendapatkan similarity scores dari film-film yang lain berdasarkan Meta_score
    meta_scores = list(enumerate(df['Meta_score']))

    # Menggabungkan similarity scores dari genre, IMDb rating, dan Meta_score
    combined_scores = [(i, 0.5 * genre_scores[i][1] + 0.3 * imdb_scores[i][1] + 0.2 * meta_scores[i][1]) for i in range(len(genre_scores))]

    # Mengurutkan film-film berdasarkan similarity scores
    combined_scores = sorted(combined_scores, key=lambda x: x[1], reverse=True)

    # Mengambil indeks film dari film-film yang paling mirip
    combined_scores = combined_scores[1:top_n+1]
    movie_indices = [i[0] for i in combined_scores]

    # Mengembalikan judul film yang direkomendasikan
    return df['title'].iloc[movie_indices]



# Contoh penggunaan
movie_title = '3 Idiots'
recommendations = get_recommendations(movie_title, cosine_sim, df)
print(f"Rekomendasi film untuk '{movie_title}':")
print(recommendations)
