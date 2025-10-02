import pandas as pd
import pickle

def print_user_movie_matrix():
    try:
        # Load the user-movie matrix
        user_movie_ratings = pickle.load(open('model/user_movie_ratings.pkl', 'rb'))

        # Check if the matrix is too large to print completely
        if user_movie_ratings.shape[0] > 20 or user_movie_ratings.shape[1] > 20:
            print(f"Matrix too large to display ({user_movie_ratings.shape[0]} users, {user_movie_ratings.shape[1]} movies).")
            print("Displaying the first 10 users and first 10 movies instead:")
            print(user_movie_ratings.iloc[:10, :10])
        else:
            print(user_movie_ratings)

    except FileNotFoundError:
        print("Error: User-movie matrix not found. Please make sure the model has been created.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    print_user_movie_matrix()
