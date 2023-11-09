import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from flask import Flask, render_template, request

app = Flask(__name__)

# Read the health-related content dataset from a file
data = pd.read_csv("health.csv")

# Read the medical conditions dataset from a file
conditions_data = pd.read_csv("condition.csv")

# Initialize the TF-IDF vectorizer for health content
tfidf_vectorizer = TfidfVectorizer()

# Create a TF-IDF matrix for the health-related dataset
tfidf_matrix = tfidf_vectorizer.fit_transform(data['Content'])

# Calculate the cosine similarity between content items
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get recommendations based on user's age, BMI, and medical conditions
def get_recommendations(age, bmi, conditions):
    age_group = data[data['Age Group'] == 'All']  # Content for all age groups
    if age >= 30:
        age_group = pd.concat([age_group, data[data['Age Group'] == '30+']])  # Content for users 30+
    if age >= 40:
        age_group = pd.concat([age_group, data[data['Age Group'] == '40+']])  # Content for users 40+
    
    indices = pd.Series(age_group.index, index=age_group['Content'])

    # Calculate cosine similarity based on user's selected content
    content_idx = indices['Exercise for a Healthy Heart']  # You can choose a different content item as a reference
    similar_scores = list(enumerate(cosine_sim[content_idx]))
    similar_scores = sorted(similar_scores, key=lambda x: x[1], reverse=True)
    similar_scores = similar_scores[1:6]  # Get the top 5 similar content items

    content_indices = [i[0] for i in similar_scores]

    recommendations = data['Content'].iloc[content_indices]

    # Create a list of recommendations
    recommendations_list = recommendations.tolist()

    # Add condition-specific recommendations
    for condition in conditions:
        condition_row = conditions_data[conditions_data['Condition'].str.lower() == condition.strip().lower()]
        if not condition_row.empty:
            recommendations_list.append(condition_row['Recommendation'].values[0])

    return recommendations_list

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_age = int(request.form['age'])
        user_weight = float(request.form['weight'])
        user_height = float(request.form['height'])
        user_bmi = user_weight / (user_height ** 2)
        user_conditions = request.form['conditions'].split(",")
        recommendations = get_recommendations(user_age, user_bmi, user_conditions)
        return render_template('Recommend.html', recommendations=recommendations)
    return render_template('index.html', recommendations=None)

if __name__ == '__main__':
    app.run(debug=True)
