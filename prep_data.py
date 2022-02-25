import os
import pandas as pd
import numpy as np
import pickle

# The code to get the bert encodings from movie metadata was from https://medium.com/analytics-vidhya/building-a-movie-recommendation-engine-in-python-53fb47547ace.  This artcile kind of sucks, but it helped me find where I can get a dataset from

movies = pd.read_csv("movies_metadata.csv", usecols=[5, 9, 20, 23])
movies["index"] = [i for i in range(0, len(movies))]
movies = movies.dropna()

vote_count = movies["vote_count"]


# This creates an embedding of a movie based on applying BERT to the description
if os.path.exists("embeds.p"):
    with open("embeds.p", "rb") as ff:
        embeds_all = np.load(ff)
        titles_all = np.load(ff)
else:
    from sentence_transformers import SentenceTransformer

    bert = SentenceTransformer("bert-base-nli-mean-tokens")
    sentence_embeddings = bert.encode(
        movies["overview"].tolist(), show_progress_bar=True
    )
    with open("embeds.p", "wb") as f:
        np.save(f, sentence_embeddings)
        np.save(f, movies["title"].tolist())

    embeds_all = sentence_embeddings
    titles_all = movies["title"].tolist()


# This creates an embedding of a movie based on user reviews.  There are ~700 people reviewing, so the "embedding" for each movie is a 1x700 vector, where the ith index is the review score from the ith person.
title_to_extid_map = {t: int(i) for t, i in zip(movies["title"], movies["id"])}
extid_to_title_map = {int(i): t for t, i in zip(movies["title"], movies["id"])}
ratings = pd.read_csv("ratings_small.csv")

titles_to_idx = {title: idx for idx, title in enumerate(titles_all)}

rating_embed = np.zeros((len(titles_all), max(ratings.userId)))
for _, r in ratings.iterrows():
    col_index = int(r.userId) - 1
    mov_id = int(r.movieId)
    if mov_id not in extid_to_title_map:
        continue
    row_index = titles_to_idx[extid_to_title_map[mov_id]]
    rating_embed[row_index, col_index] = r.rating

# Save the data
data_to_dump = {
    "vote_count": np.float64(vote_count.tolist()),
    "rating_embed": rating_embed,
    "bert_embed": embeds_all,
    "titles": titles_all,
}

with open("data.p", "wb") as p:
    pickle.dump(data_to_dump, p)
