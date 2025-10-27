import pandas as pd
import numpy as np
import pickle
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds

print("="*70)
print("BUILDING RECOMMENDATION MODELS FROM SCRATCH")
print("="*70)

# Step 1: Load and merge movie data
print("\n[1/6] Loading movie data...")
movies = pd.read_csv('movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
movies_tmdb = pd.read_csv('tmdb_5000_movies.csv')

print(f"   ✓ Movies: {movies.shape}")
print(f"   ✓ Credits: {credits.shape}")
print(f"   ✓ TMDB Movies: {movies_tmdb.shape}")

# Merge datasets
movies_tmdb = movies_tmdb.merge(credits, left_on='id', right_on='movie_id', how='left')
print(f"   ✓ Merged dataset: {movies_tmdb.shape}")

# Step 2: Process features for content-based filtering
print("\n[2/6] Processing movie features...")

def safe_eval(x):
    """Safely evaluate string to list"""
    try:
        return ast.literal_eval(x)
    except:
        return []

def get_names(x):
    """Extract names from list of dicts"""
    if isinstance(x, list):
        return [i['name'] for i in x if isinstance(i, dict) and 'name' in i]
    return []

def get_director(x):
    """Extract director from crew"""
    if isinstance(x, list):
        for i in x:
            if isinstance(i, dict) and i.get('job') == 'Director':
                return [i['name']]
    return []

# Process genres
if 'genres' in movies_tmdb.columns:
    movies_tmdb['genres'] = movies_tmdb['genres'].apply(safe_eval).apply(get_names)
else:
    movies_tmdb['genres'] = ''

# Process keywords
if 'keywords' in movies_tmdb.columns:
    movies_tmdb['keywords'] = movies_tmdb['keywords'].apply(safe_eval).apply(get_names)
else:
    movies_tmdb['keywords'] = ''

# Process cast (top 3 actors)
if 'cast' in movies_tmdb.columns:
    movies_tmdb['cast'] = movies_tmdb['cast'].apply(safe_eval).apply(lambda x: get_names(x)[:3])
else:
    movies_tmdb['cast'] = ''

# Process crew (director)
if 'crew' in movies_tmdb.columns:
    movies_tmdb['director'] = movies_tmdb['crew'].apply(safe_eval).apply(get_director)
else:
    movies_tmdb['director'] = ''

# Create tags by combining all features
def create_tags(row):
    """Combine all features into tags"""
    tags = []
    
    # Add genres
    if isinstance(row.get('genres'), list):
        tags.extend([str(x).replace(" ", "") for x in row['genres']])
    
    # Add keywords
    if isinstance(row.get('keywords'), list):
        tags.extend([str(x).replace(" ", "") for x in row['keywords']])
    
    # Add cast
    if isinstance(row.get('cast'), list):
        tags.extend([str(x).replace(" ", "") for x in row['cast']])
    
    # Add director
    if isinstance(row.get('director'), list):
        tags.extend([str(x).replace(" ", "") for x in row['director']])
    
    return ' '.join(tags).lower()

movies_tmdb['tags'] = movies_tmdb.apply(create_tags, axis=1)

# Prepare final movie list
movie_list = movies[['movieId', 'title', 'genres']].copy()
print(f"   ✓ Final movie list: {movie_list.shape}")

# Step 3: Create content-based similarity matrix
print("\n[3/6] Creating content-based similarity matrix...")
print("   This may take a few minutes...")

# Use only movies that have tags
movies_with_tags = movies_tmdb[movies_tmdb['tags'].str.len() > 0].copy()
print(f"   ✓ Movies with features: {len(movies_with_tags)}")

# Create feature vectors using CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies_with_tags['tags']).toarray()
print(f"   ✓ Feature vectors shape: {vectors.shape}")

# Calculate cosine similarity
similarity = cosine_similarity(vectors)
print(f"   ✓ Similarity matrix shape: {similarity.shape}")

# Save content similarity
pickle.dump(similarity, open('model/content_similarity.pkl', 'wb'))
print("   ✓ Saved: model/content_similarity.pkl")

# Step 4: Create collaborative filtering model
print("\n[4/6] Creating collaborative filtering model...")
ratings = pd.read_csv('ratings.csv')
print(f"   ✓ Ratings loaded: {ratings.shape}")

# Create user-movie matrix
user_movie_ratings = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
print(f"   ✓ User-movie matrix: {user_movie_ratings.shape}")

# Perform SVD (Matrix Factorization)
print("   ✓ Performing SVD (this may take a minute)...")
k = 50  # Number of latent factors
U, sigma, Vt = svds(user_movie_ratings.values, k=k)
sigma = np.diag(sigma)

# Predict all ratings
predicted_ratings = np.dot(np.dot(U, sigma), Vt)
print(f"   ✓ Predicted ratings shape: {predicted_ratings.shape}")

# Save collaborative model
pickle.dump(predicted_ratings, open('model/collaborative_predictions.pkl', 'wb'))
print("   ✓ Saved: model/collaborative_predictions.pkl")

# Step 5: Save supporting data
print("\n[5/6] Saving supporting data...")

pickle.dump(movie_list, open('model/movie_list.pkl', 'wb'))
print("   ✓ Saved: model/movie_list.pkl")

pickle.dump(user_movie_ratings, open('model/user_movie_ratings.pkl', 'wb'))
print("   ✓ Saved: model/user_movie_ratings.pkl")

pickle.dump(ratings, open('model/ratings_df.pkl', 'wb'))
print("   ✓ Saved: model/ratings_df.pkl")

# Step 6: Verify models
print("\n[6/6] Verifying models...")
print("   Testing Iron Man (2008) recommendations...")

# Find Iron Man in TMDB data
iron_man_idx = movies_tmdb[movies_tmdb['title_x'] == 'Iron Man'].index
if len(iron_man_idx) > 0:
    iron_man_idx = iron_man_idx[0]
    
    # Get top 5 similar movies
    distances = similarity[iron_man_idx]
    similar_indices = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    print("\n   Top 5 similar movies to Iron Man:")
    for i, (idx, score) in enumerate(similar_indices, 1):
        title = movies_tmdb.iloc[idx]['title_x']
        print(f"   {i}. {title} (similarity: {score:.4f})")
else:
    print("   ⚠ Iron Man not found in TMDB data for testing")

print("\n" + "="*70)
print("✅ MODEL CREATION COMPLETE!")
print("="*70)
print("\nAll model files saved in 'model/' directory:")
print("  • content_similarity.pkl")
print("  • collaborative_predictions.pkl")
print("  • movie_list.pkl")
print("  • user_movie_ratings.pkl")
print("  • ratings_df.pkl")
print("\nYou can now run: py -m streamlit run app.py")
