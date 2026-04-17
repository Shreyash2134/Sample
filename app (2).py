
import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# NOTE: In a real standalone deployment, you would save and load the encoders.
# For this Colab environment, we assume 'df_cleaned' was available globally during development.
# For a true standalone app, you would need to save these mappings or a dummy dataframe
# to fit the encoders with, ensuring consistency.

# --- Placeholder for recreating encoders (if df_cleaned not available standalone) ---
# In a production setting, you'd load pre-fitted encoders or a minimal dataset
# that contains all unique values seen during training.
# For this demonstration, we'll assume the original dataframe structure is known.

# Example of how you might re-initialize and fit encoders if running standalone
# This assumes you know all possible categories from training data.
known_genders = ['Male', 'Female'] # Replace with actual unique genders from training
known_education_levels = ["Bachelor's", "Master's", "PhD"] # Replace with actual unique education levels
known_job_titles = ['Software Engineer', 'Data Analyst', 'Senior Manager', 'Sales Associate', 'Director'] # etc.

le_gender = LabelEncoder()
le_education = LabelEncoder()
le_job_title = LabelEncoder()

# Fit with known categories that were used during model training
le_gender.fit(known_genders)
le_education.fit(known_education_levels)
# NOTE: For Job Title, this list needs to be exhaustive of all job titles seen during training
# A more robust solution would be to save the fitted LabelEncoder objects directly.
# For this example, we'll fit with a sample to avoid errors, but be aware of limitations.
# A robust solution would involve saving the entire set of unique job titles.
# For the purpose of this example, we will use a small sample and note the limitation.
le_job_title.fit(known_job_titles)
# --- End Placeholder ---

# Load the trained Random Forest model
model_filename = 'random_forest_salary_predictor.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Streamlit App Title
st.title('Salary Predictor')
st.write('Enter the employee details to predict their salary.')

# Input fields for user
age = st.slider('Age', min_value=18, max_value=65, value=30)
years_of_experience = st.slider('Years of Experience', min_value=0, max_value=40, value=5)

gender_options = known_genders # Use known categories
gender = st.selectbox('Gender', options=gender_options)

education_options = known_education_levels # Use known categories
education_level = st.selectbox('Education Level', options=education_options)

# This part is highly dependent on having all job titles. For a simple demo, it's fine,
# but for production, ensure 'known_job_titles' is comprehensive.
job_title_options = known_job_titles # Use known categories
job_title = st.selectbox('Job Title', options=job_title_options)

# Predict button
if st.button('Predict Salary'):
    try:
        # Encode categorical inputs
        gender_encoded = le_gender.transform([gender])[0]
        education_encoded = le_education.transform([education_level])[0]
        # Ensure all job titles are handled, if a new one appears, this will fail.
        job_title_encoded = le_job_title.transform([job_title])[0]

        # Create a DataFrame for prediction
        input_data = pd.DataFrame([[age, years_of_experience, gender_encoded, education_encoded, job_title_encoded]],
                                  columns=['Age', 'Years of Experience', 'Gender_Encoded', 'Education Level_Encoded', 'Job Title_Encoded'])

        # Make prediction
        predicted_salary = model.predict(input_data)[0]

        st.success(f'Predicted Salary: ${predicted_salary:,.2f}')
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
