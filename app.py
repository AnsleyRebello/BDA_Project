import pandas as pd
import numpy as np
import pickle
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from scipy.sparse.linalg import svds
import streamlit as st
import os

# Previous helper functions remain the same
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L 

def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 

def collapse(L):
    return [i.replace(" ", "") for i in L]

# Model creation functions removed - models should be pre-generated



def get_hybrid_recommendations(movie, user_id=None, content_weight=0.7, collab_weight=0.3, num_recommendations=5):
    """Get hybrid recommendations"""
    try:
        # Load models and data
        movies = pickle.load(open('model/movie_list.pkl', 'rb'))
        content_similarity = pickle.load(open('model/content_similarity.pkl', 'rb'))
        collaborative_predictions = pickle.load(open('model/collaborative_predictions.pkl', 'rb'))
        user_movie_ratings = pickle.load(open('model/user_movie_ratings.pkl', 'rb'))
        
        # Load TMDB data for matching
        movies_tmdb = pd.read_csv('tmdb_5000_movies.csv')
        credits = pd.read_csv('tmdb_5000_credits.csv')
        movies_tmdb = movies_tmdb.merge(credits, left_on='id', right_on='movie_id', how='left')
        
        # Verify movie exists in TMDB data
        movie_matches = movies_tmdb[movies_tmdb['title_x'].str.contains(movie.split('(')[0].strip(), case=False, na=False)]
        
        if len(movie_matches) == 0:
            return f"Error: Movie '{movie}' not found in database."
        
        # Get the movie index
        movie_idx = movie_matches.index[0]
        
        # Get content-based similarities
        content_scores = content_similarity[movie_idx]
        
        # Get collaborative scores if user_id provided
        collab_scores = None
        if user_id is not None and user_id in user_movie_ratings.index:
            user_idx = user_movie_ratings.index.get_loc(user_id)
            if user_idx < len(collaborative_predictions):
                collab_scores = collaborative_predictions[user_idx]
                collab_weight = 0.3
            else:
                collab_weight = 0
                content_weight = 1.0
        else:
            collab_weight = 0
            content_weight = 1.0
        
        # Calculate hybrid scores
        hybrid_scores = []
        for idx in range(len(content_scores)):
            content_score = content_scores[idx]
            
            # Get collaborative score if available
            if collab_scores is not None:
                # Map TMDB index to movieId
                tmdb_movie_id = movies_tmdb.iloc[idx]['id']
                if tmdb_movie_id in user_movie_ratings.columns:
                    col_idx = user_movie_ratings.columns.get_loc(tmdb_movie_id)
                    collab_score = collab_scores[col_idx] if col_idx < len(collab_scores) else 0
                    collab_score_norm = (collab_score - 0) / (5 - 0)
                else:
                    collab_score = 0
                    collab_score_norm = 0
            else:
                collab_score = 0
                collab_score_norm = 0
            
            # Calculate hybrid score
            hybrid_score = (content_weight * content_score) + (collab_weight * collab_score_norm)
            hybrid_scores.append((idx, hybrid_score, content_score, collab_score))
        
        # Sort by hybrid score (descending)
        hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)
        
        # Get top recommendations
        recommendations = []
        count = 0
        seen_titles = set()
        
        for idx, hybrid_score, content_score, collab_score in hybrid_scores:
            if count >= num_recommendations:
                break
            
            title = movies_tmdb.iloc[idx]['title_x']
            year = movies_tmdb.iloc[idx]['release_date'][:4] if pd.notna(movies_tmdb.iloc[idx]['release_date']) else ''
            full_title = f"{title} ({year})" if year else title
            
            # Skip the same movie and duplicates
            if title != movie.split('(')[0].strip() and title not in seen_titles:
                seen_titles.add(title)
                recommendations.append({
                    'title': full_title,
                    'similarity': round(hybrid_score * 100, 2),
                    'content_score': round(content_score * 100, 2),
                    'collab_score': round(collab_score * 20, 2) if collab_weight > 0 else 0
                })
                count += 1
        
        return recommendations
    
    except FileNotFoundError as e:
        return f"Error: Model files not found. Please run create_models.py first. ({str(e)})"
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    st.set_page_config(
        page_title="Cinematic Matchmaker", 
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for modern dark theme
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a1a 100%);
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #00c9ff 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0, 201, 255, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .main-header h1 {
        color: white !important;
        font-size: 3rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9) !important;
        font-size: 1.2rem !important;
        font-weight: 400 !important;
        margin: 0 !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 0.5rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 10px;
        color: #ffffff;
        font-weight: 500;
        padding: 0.75rem 1.5rem;
        margin: 0 0.25rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(0, 201, 255, 0.2);
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00c9ff, #92fe9d) !important;
        color: #000000 !important;
        font-weight: 600 !important;
        box-shadow: 0 5px 15px rgba(0, 201, 255, 0.4);
    }
    
    .content-card {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(0, 201, 255, 0.3) !important;
        border-radius: 10px !important;
        color: white !important;
    }
    
    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(0, 201, 255, 0.3) !important;
        border-radius: 10px !important;
        color: white !important;
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #00c9ff, #92fe9d) !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 15px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6) !important;
    }
    
    .movie-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05));
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(0, 201, 255, 0.2);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .movie-card:hover {
        transform: translateY(-5px);
        border-color: rgba(0, 201, 255, 0.5);
        box-shadow: 0 10px 30px rgba(0, 201, 255, 0.3);
    }
    
    .movie-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #00c9ff, #92fe9d);
    }
    
    .movie-title {
        color: #ffffff !important;
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.5rem !important;
    }
    
    .similarity-badge {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        display: inline-block;
        margin-top: 0.5rem;
    }
    
    .stSuccess {
        background: linear-gradient(135deg, rgba(46, 213, 115, 0.1), rgba(0, 201, 255, 0.1)) !important;
        border: 1px solid rgba(46, 213, 115, 0.3) !important;
        border-radius: 15px !important;
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.1), rgba(238, 90, 36, 0.1)) !important;
        border: 1px solid rgba(255, 107, 107, 0.3) !important;
        border-radius: 15px !important;
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(255, 159, 67, 0.1), rgba(255, 206, 84, 0.1)) !important;
        border: 1px solid rgba(255, 159, 67, 0.3) !important;
        border-radius: 15px !important;
    }
    
    .sidebar-content {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(0, 201, 255, 0.1), rgba(146, 254, 157, 0.1));
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(0, 201, 255, 0.2);
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    p, span, div {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    .stMarkdown {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Loading spinner styling */
    .stSpinner > div {
        border-top-color: #00c9ff !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #00c9ff, #92fe9d);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #0099cc, #7acc7a);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header with modern styling
    st.markdown("""
    <div class="main-header">
        <h1>🎬 Cinematic Matchmaker</h1>
        <p>Advanced AI-powered movie matching using hybrid recommendation algorithms</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with modern design
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.markdown("### 🎛️ Control Panel")
        
        # App info
        st.markdown("#### ℹ️ About")
        st.markdown("""
        **Cinematic Matchmaker** uses advanced AI algorithms to find your perfect movie matches.
        
        🔬 **Technologies:**
        - Content-based filtering
        - Collaborative filtering  
        - Hybrid recommendation system
        - Machine learning algorithms
        """)
        
        # Statistics (if model exists)
        try:
            movies = pickle.load(open('model/movie_list.pkl', 'rb'))
            user_ratings = pickle.load(open('model/user_movie_ratings.pkl', 'rb'))
            
            st.markdown("#### 📊 Database Stats")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin: 0; color: #00c9ff;">{len(movies):,}</h3>
                    <p style="margin: 0; font-size: 0.8rem;">Movies</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin: 0; color: #92fe9d;">{len(user_ratings):,}</h3>
                    <p style="margin: 0; font-size: 0.8rem;">Users</p>
                </div>
                """, unsafe_allow_html=True)
                
        except FileNotFoundError:
            st.markdown("#### ⚠️ Model Required")
            st.warning("Model files not found! Please ensure model files are available in the 'model' directory.")
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; opacity: 0.7;">
            <small>Made with ❤️ using Streamlit</small>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content area for movie matches
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("### 🎯 Get Movie Matches")
    st.markdown("Discover your perfect movie matches using our AI algorithm")
    
    try:
        # Load TMDB movies (the ones that have similarity data)
        movies_tmdb = pd.read_csv('tmdb_5000_movies.csv')
        credits = pd.read_csv('tmdb_5000_credits.csv')
        movies_tmdb = movies_tmdb.merge(credits, left_on='id', right_on='movie_id', how='left')
        
        # Create movie list with year
        movie_titles = []
        for _, row in movies_tmdb.iterrows():
            title = row['title_x']
            year = row['release_date'][:4] if pd.notna(row['release_date']) else ''
            full_title = f"{title} ({year})" if year else title
            movie_titles.append(full_title)
        
        movie_titles = sorted(list(set(movie_titles)))  # Remove duplicates and sort
        
        # Create modern input layout
        st.markdown("#### 🎬 Select Movie")
        selected_movie = st.selectbox(
            "Choose a movie you like:",
            movie_titles,
            help="Select a movie to find similar matches"
        )
        
        # Weight adjustment section
        st.markdown("#### ⚖️ Adjust Matchmaking Algorithm")
        st.markdown("Fine-tune how our AI finds your matches:")
        
        col1, col2 = st.columns(2)
        with col1:
            content_weight = st.slider(
                "🎭 Content-based Weight",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Higher values focus on movie characteristics (genre, cast, etc.)"
            )
        with col2:
            collab_weight = 1 - content_weight
            st.slider(
                "👥 Collaborative Weight",
                min_value=0.0,
                max_value=1.0,
                value=collab_weight,
                step=0.1,
                disabled=True,
                help="Automatically calculated as 1 - Content Weight"
            )
        
        # Modern button with centered layout
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button('🔍 Find My Matches', key="find_matches"):
                with st.spinner("🎯 Finding your perfect matches..."):
                    recommendations = get_hybrid_recommendations(
                        selected_movie,
                        None,  # No user_id
                        content_weight,
                        collab_weight
                    )
                    
                    if isinstance(recommendations, list):
                        st.markdown("### 🌟 Your Matched Movies")
                        
                        for i, rec in enumerate(recommendations, 1):
                            st.markdown(f"""
                            <div class="movie-card">
                                <div class="movie-title">#{i} {rec['title']}</div>
                                <div style="display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px;">
                                    <span class="similarity-badge">Match: {rec['similarity']}%</span>
                                    <span style="background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.85rem;">
                                        Content: {rec['content_score']}%
                                    </span>
                                    <span style="background: linear-gradient(135deg, #f093fb, #f5576c); color: white; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.85rem;">
                                        Collaborative: {rec['collab_score']}%
                                    </span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.error(f"❌ {recommendations}")
        
        st.markdown('</div>', unsafe_allow_html=True)
                    
    except FileNotFoundError:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.error("❌ Model files not found. Please ensure the model files exist in the 'model' directory.")
        st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.error(f"❌ Error: {str(e)}")
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()