import csv
import datetime
import webbrowser
import numpy as np
from flask import Flask, render_template, request, jsonify, session
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = 'your_secret_key'

complaints_csv_path = r'C:\Users\Amith kumar\Desktop\PROJECTS\CUSTOMER CHATBOT\ola_chatbot\src\complaints.csv'
bookings_csv_path = r'C:\Users\Amith kumar\Desktop\PROJECTS\CUSTOMER CHATBOT\ola_chatbot\src\bookings.csv'

# Predefined responses and options
responses = {
    "about": "Welcome to X-Cabs! X-Cabs is a reliable and efficient ride-sharing service offering convenient and affordable transportation.",
    "raise a complaint": "Please tell us your complaint. We will get back to you soon.",
    "book a ride": "Please provide your current location, destination, and preferred departure time, and we'll confirm your cab booking.",
    "talk to an agent": "One of our agents will contact you at the registered phone number."
}

options = ["about", "raise a complaint", "book a ride", "talk to an agent"]

training_phrases = {
    "about": ["about", "tell me about X-Cabs", "what is X-Cabs", "describe X-Cabs", "information about X-Cabs"],
    "raise a complaint": ["raise a complaint", "file a complaint", "complaint", "issue", "report a problem"],
    "book a ride": ["book a ride", "book a cab", "ride booking", "cab booking", "book a car"],
    "talk to an agent": ["talk to an agent", "speak to an agent", "speak to support", "agent", "contact support"],
    "restart": ["restart", "start over", "start a new conversation", "clear"]
}

vectorizer = TfidfVectorizer()
complaint = []
booking = []

# Function to find the best match
def get_best_match(user_input):
    all_phrases = sum(training_phrases.values(), [])
    vectors = vectorizer.fit_transform(all_phrases + [user_input])
    cosine_similarities = cosine_similarity(vectors[-1], vectors[:-1])
    best_match_index = np.argmax(cosine_similarities)

    cumulative_phrases_count = 0
    for option, phrases in training_phrases.items():
        if cumulative_phrases_count <= best_match_index < cumulative_phrases_count + len(phrases):
            return option
        cumulative_phrases_count += len(phrases)
    return None

def complaint_to_csv(complaints):
    with open(complaints_csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write a new row with current datetime, complaints, and "pending"
        writer.writerow([datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ', '.join(complaints), "pending"])

def booking_to_csv(source, destination, departure_time):
    with open(bookings_csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write a new row with current datetime, source, destination, and departure time
        writer.writerow([datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), source, destination, departure_time])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    user_input = request.form.get('user_input', '').lower()

    # Handle complaint mode
    if 'complaint_mode' in session and session['complaint_mode']:
        if user_input == "done":
            session['complaint_mode'] = False
            complaint_to_csv(complaint)
            complaint.clear()  # Clear the list for future complaints
            return f"Your complaints have been registered: \n We will get back to you soon."
        else:
            complaint.append(user_input)
            return "Got it! Anything else you'd like to add? Type 'done' when finished."

    # Handle booking mode
    if 'booking_mode' in session and session['booking_mode']:
        booking_stage = session.get('booking_stage', 0)

        if booking_stage == 0:
            booking.append(user_input)
            session['booking_stage'] = 1
            return "Where would you like to go? (Destination)"
        elif booking_stage == 1:
            booking.append(user_input)
            session['booking_stage'] = 2
            return "When would you like to leave? (Departure time)"
        elif booking_stage == 2:
            booking.append(user_input)
            session['booking_mode'] = False
            session.pop('booking_stage', None)
            booking_to_csv(booking[0], booking[1], booking[2])
            booking_details = f"From: {booking[0]}, To: {booking[1]}, At: {booking[2]}"
            booking.clear()  # Clear the list for future bookings
            return f"Your booking has been confirmed: {booking_details}"

    # Handle general chatbot interaction
    matched_option = get_best_match(user_input)
    if matched_option:
        if matched_option == "about":
            return responses["about"]
        elif matched_option == "raise a complaint":
            session['complaint_mode'] = True
            return responses["raise a complaint"]
        elif matched_option == "book a ride":
            session['booking_mode'] = True
            session['booking_stage'] = 0
            booking.clear()  # Ensure the booking list is empty
            return "Where are you starting from? (Source)"
        elif matched_option == "talk to an agent":
            return responses["talk to an agent"]
        elif matched_option == "restart":
            session.pop('complaint_mode', None)
            session.pop('booking_mode', None)
            session.pop('booking_stage', None)
            
            webbrowser.open("http://localhost:5000/")
            return "Restarting the conversation. Please choose an option or ask your question again."
    return "I'm sorry, I didn't understand that. Could you please rephrase your question?"


if __name__ == '__main__':
    app.run(debug=True)
