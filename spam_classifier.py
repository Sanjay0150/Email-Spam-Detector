import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# A larger dataset for spam detection (for demonstration purposes)
data = [
    ("Free money available now!", 1),  # Spam
    ("Hello, how are you?", 0),        # Not Spam
    ("Congratulations, you've won a prize!", 1),  # Spam
    ("Let's catch up soon", 0),         # Not Spam
    ("Limited time offer! Get your free gift now!", 1),  # Spam
    ("Important: Your account has been updated.", 0),   # Not Spam
    ("Act fast! Only a few days left to claim your prize.", 1),  # Spam
    ("Reminder: Your meeting is scheduled for 3 PM tomorrow.", 0),  # Not Spam
    ("Get rich quick! Invest now and earn huge returns!", 1),  # Spam
    ("Please review the attached document.", 0),  # Not Spam
    ("You have won a lottery! Claim your cash prize today!", 1)  # Spam
]

# Prepare the data for training
texts, labels = zip(*data)

# TF-IDF Vectorizer with more features (5000 features instead of 5)
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(texts).toarray()  # Convert text to TF-IDF features
y = labels

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X, y)

# Function to classify text as spam or not spam
def classify_text(text):
    transformed_text = tfidf.transform([text]).toarray()
    prediction = model.predict(transformed_text)
    return "Spam" if prediction[0] == 1 else "Not Spam"

# Streamlit UI
st.title("Improved Email Spam Classifier")

# Allow user to upload an image
uploaded_image = st.file_uploader("Upload an image of an email", type=["png", "jpg", "jpeg"])

# Let the user manually input the text extracted from the image
text_input = st.text_area("Enter the extracted text from the image")

# Add a submit button that only works when the image is uploaded and text is entered
if uploaded_image and text_input:
    submit_button = st.button("Submit")
    
    if submit_button:
        # Show the uploaded image to the user
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        
        # Classify the entered text as spam or not spam
        result = classify_text(text_input)
        st.write(f"The email is classified as: **{result}**")
