import numpy as np
import csv
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Predefined responses and options
responses = {
    "about": "Welcome to X-Cabs! X-Cabs is a reliable and efficient ride-sharing service offering convenient and affordable transportation.",
    "raise a complaint": "Please tell us your complaint, and it has been registered. We will look into it and get back to you soon.",
    "book a ride": "Please provide your current location, destination, and preferred departure time, and we'll confirm your cab booking.",
    "talk to an agent": "Please share your complaint, and our agent will contact you at the registered phone number."
}

# Define the possible options
options = ["about", "raise a complaint", "book a ride", "talk to an agent"]

# Training phrases for similarity comparison
training_phrases = {
    "about": ["about", "tell me about X-Cabs", "what is X-Cabs", "describe X-Cabs", "information about X-Cabs"],
    "raise a complaint": ["raise a complaint", "file a complaint", "complaint", "issue", "report a problem"],
    "book a ride": ["book a ride", "book a cab", "ride booking", "cab booking", "book a car"],
    "talk to an agent": ["talk to an agent", "speak to an agent", "speak to support", "agent", "contact support"]
}

# Initialize the vectorizer
vectorizer = TfidfVectorizer()

# Function to compute cosine similarity and find the closest match
def get_best_match(user_input):
    # Combine all training phrases into a single list
    all_phrases = sum(training_phrases.values(), [])
    
    # Fit the vectorizer to the training phrases
    vectors = vectorizer.fit_transform(all_phrases + [user_input])  # Add the user input at the end
    
    # Compute cosine similarities between the user input and all training phrases
    cosine_similarities = cosine_similarity(vectors[-1], vectors[:-1])
    
    # Find the index of the most similar phrase
    best_match_index = np.argmax(cosine_similarities)
    
    # Find the corresponding option for the best match
    cumulative_phrases_count = 0
    for option, phrases in training_phrases.items():
        # Check if the best match falls within the range of the current option's phrases
        if cumulative_phrases_count <= best_match_index < cumulative_phrases_count + len(phrases):
            return option
        cumulative_phrases_count += len(phrases)
    
    return None

# Function to write complaints to the CSV file
def log_complaint(complaint_text):
    with open('complaints.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([timestamp, complaint_text, "pending"])

# Function to write bookings to the CSV file
def log_booking(source, destination, departure_time):
    with open('bookings.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([timestamp, source, destination, departure_time])

# Main function for the chatbot interaction
def start_chat():
    print("Welcome to X-Cabs! How can I help you today?")
    print("1. About X-Cabs")
    print("2. Raise a Complaint")
    print("3. Book a Ride")
    print("4. Talk to an Agent")
    
    while True:
        user_input = input("\nPlease choose an option or type your question: ").lower()
        
        # Get the best matching option based on user input
        matched_option = get_best_match(user_input)
        
        if matched_option:
            if matched_option == "about":
                print(responses["about"])
            elif matched_option == "raise a complaint":
                complaint = input("Please tell us your complaint: ")
                log_complaint(complaint)  # Log the complaint to the CSV
                print("Your complaint has been registered. We will look into it and get back to you soon.")
            elif matched_option == "book a ride":
                source = input("Please provide your current location: ")
                destination = input("Please provide your destination: ")
                departure_time = input("Please provide your preferred departure time: ")
                log_booking(source, destination, departure_time)  # Log the booking to the CSV
                print(f"Your cab has been confirmed from {source} to {destination} at {departure_time}.")
            elif matched_option == "talk to an agent":
                complaint = input("Please tell us your complaint: ")
                log_complaint(complaint)  # Log the complaint to the CSV
                print("Our agent will contact you on your registered phone number.")
        else:
            print("I'm sorry, I didn't understand that. Could you please choose one of the options or rephrase your question?")
        
        continue_chat = input("\nWould you like to do something else? (yes/no): ").lower()
        if continue_chat != "yes":
            print("Thank you for using X-Cabs! Have a great day!")
            break

if __name__ == "__main__":
    # Initialize CSV files with headers if they don't already exist
    try:
        with open('complaints.csv', mode='x', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['date', 'complaint', 'status'])  # Writing header to complaints.csv
    except FileExistsError:
        pass
    
    try:
        with open('bookings.csv', mode='x', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['date', 'source', 'destination', 'departure_time'])  # Writing header to bookings.csv
    except FileExistsError:
        pass

    start_chat()
