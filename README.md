# Movie Recommendation System


## Project Overview

This project implements a comprehensive movie recommendation system that provides personalized movie suggestions based on user preferences. It utilizes various methodologies, including collaborative filtering, autoencoders, deep learning, and transformer architectures, to enhance recommendation accuracy and user experience.

## Table of Contents
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Methodologies](#methodologies)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Technologies Used

- **Programming Languages:** Python
- **Libraries:** 
  - TensorFlow
  - scikit-learn
  - Pandas
  - NumPy
  - Matplotlib
  - FuzzyWuzzy
  - OpenAI API (if applicable)
- **Data Storage:** Pandas DataFrames
- **Modeling Techniques:** 
  - K-Nearest Neighbors (KNN)
  - Autoencoders
  - Neural Collaborative Filtering (NCF)
  - Transformer Models

## Dataset

The project utilizes the [MovieLens dataset](https://grouplens.org/datasets/movielens/) for training and testing the recommendation models. The dataset contains user ratings for a wide range of movies, which is used to create user-item interaction matrices.

## Key Features

- **Personalized Recommendations:** Provides tailored movie suggestions based on user preferences and historical data.
- **Multiple Recommendation Approaches:** Implements KNN, autoencoders, NCF, and transformer models to enhance recommendation accuracy.
- **Performance Evaluation:** Evaluates model performance using precision and recall metrics, visualizing results for easy comparison.
- **User-friendly Interface:** (If applicable) Created a simple interface for users to interact with the recommendation system.

## Project Structure


|           File                      | Description                                    |
|-------------------------------------|------------------------------------------------|
| `movie-recommendation-system/`      | Root directory for the project                |
| `data/`                             | Contains dataset files                        |
| ├── `movies.csv`                   | Movie metadata                                 |
| └── `ratings.csv`                  | User ratings data                             |
| `notebooks/`                       | Jupyter notebooks for exploratory data analysis|
| └── `data_exploration.ipynb`       | Data exploration notebook                      |
| `models/`                          | Model implementations                          |
| ├── `knn_model.py`                 | K-Nearest Neighbors model                     |
| ├── `autoencoder_model.py`          | Autoencoder model                             |
| ├── `ncf_model.py`                 | Neural Collaborative Filtering model           |
| └── `transformer_model.py`          | Transformer-based model                       |
| `README.md`                        | Project documentation      



## Methodologies

1. **Data Processing:**
   - Loaded and cleaned the MovieLens dataset to create user-item interaction matrices.
   - Handled missing values and transformed the data to ensure compatibility with different models.

2. **Modeling Approaches:**
   - **K-Nearest Neighbors (KNN):** Implemented KNN to identify similar users and items based on ratings, achieving baseline metrics of precision 0.5 and recall 0.1.
   - **Autoencoders:** Developed autoencoder models to learn user and item embeddings, capturing latent features and improving recommendation quality.
   - **Neural Collaborative Filtering (NCF):** Designed and trained an NCF model to capture complex interactions between users and items, enhancing recommendation accuracy.
   - **Transformer Model:** Leveraged transformer architectures to utilize attention mechanisms for improved modeling of user preferences.

3. **Evaluation:**
   - Assessed the performance of each model using precision and recall metrics.
   - Visualized the results using Matplotlib to compare the effectiveness of different methodologies and inform optimizations.
  



    ## Graph for data
   ### sample for movies.csv

  ![image](https://github.com/user-attachments/assets/bfad25de-cf41-4675-a81f-3fae213dcaca)




  ![image](https://github.com/user-attachments/assets/4e7d2e12-1ab1-42ff-8a14-f33dd835e159)


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Vikramjeetsingh07/Movie-Recomendation-system-using-KNN-Autoencoders-NCF-transformers
   cd movie-recommendation

Usage
Load the dataset and preprocess it:

python
Copy code
import pandas as pd
from your_preprocessing_module import preprocess_data

movies = pd.read_csv('data/movies.csv')
ratings = pd.read_csv('data/ratings.csv')
user_item_matrix = preprocess_data(movies, ratings)
Train and evaluate models:

python
Copy code
from your_model_module import train_knn_model, train_autoencoder_model, train_ncf_model, train_transformer_model

knn_model = train_knn_model(user_item_matrix)
autoencoder_model = train_autoencoder_model(user_item_matrix)
ncf_model = train_ncf_model(user_item_matrix)
transformer_model = train_transformer_model(user_item_matrix)
Generate recommendations:

python
Copy code
recommendations = knn_model.recommend(user_id)
Evaluation
The models are evaluated using precision and recall metrics. Results are visualized to assess the strengths and weaknesses of each approach, leading to informed optimizations.

Results
Autoencoders: Precision:- 0.54
Neural Collaborative Filtering: Precision:- 0.75
Transformer Model: Precision:- 0.73
![image](https://github.com/user-attachments/assets/727ceadc-6d95-457c-a4da-70181d1f7729)

![image](https://github.com/user-attachments/assets/0225b9c4-ccb6-455b-b791-d9969b6d7d7d)

![image](https://github.com/user-attachments/assets/40721411-6187-4047-bc54-3ee99a5b3dde)




Contributing
Contributions are welcome! Please feel free to submit a pull request or raise an issue to discuss improvements and enhancements.

License
This project is licensed under the MIT License - see the LICENSE file for details.

