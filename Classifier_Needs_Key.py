from skllm.config import SKLLMConfig

SKLLMConfig.set_openai_key("USER KEY CAN BE FOUND ON OPENAI")
SKLLMConfig.set_openai_org("USER ORG CAN BE FOUND ON OPENAI")

import numpy as np
from skllm import FewShotGPTClassifier

# Define your training data. Replace these with your actual data.
X_train = [
    # Sex trafficking examples
    "I am an open-minded, Spinner",
    "Companionship services, fresh faces from abroad available for various tasks",
    "Exclusive opportunity to meet teens and young adults for confidential encounters",
    "NEW in Town ✈ , Babyback, Princess, Petite, Young, in the Circuit, Spinner, in Training, Wet Behind the Ears, Crash-Test Dummy, SYT, TOFTT",
    
    # Examples NOT related to trafficking
    "My desire is to have you enjoy parts of my body; I make sure you can't help but come back for more",
    "Rock your world and satisfy you in way that you only thought were possible in your deepest fantasies",
    "I am here to ensure you enjoy the entire stimulating and sensual journey, and not merely the destination",
    "My ability to exceed your every expectation is one of the reasons why I'm in such demand"
]

y_train = [
    "trafficking",
    "trafficking",
    "trafficking",
    "trafficking",
    "not trafficking",
    "not trafficking",
    "not trafficking",
    "not trafficking"
]

# Initialize and train the classifier
clf = FewShotGPTClassifier(openai_model="gpt-3.5-turbo")  # Replace with your actual model name
clf.fit(X_train, y_train)

# Define your test data (new instances that are unseen during training)
X_test = [
    # Sex trafficking examples
    "New arrivals just in, available for private entertainment, satisfaction assured",
    "Meet young and adventurous companions, discreet fun, no commitments",
    
    # Examples NOT related to trafficking
    "Cleaning service hiring, flexible hours, good pay",
    "Locals looking for delivery drivers, friendly work environment"
]

y_test = [
    "trafficking",
    "trafficking",
    "not trafficking",
    "not trafficking"
]

# Predict the labels of the test data and check each prediction for accuracy
correct_predictions = 0
for index, text in enumerate(X_test):
    prediction = clf.predict([text])[0]
    correctness = "Correct" if prediction == y_test[index] else "Incorrect"
    if correctness == "Correct":
        correct_predictions += 1
    
    print(f"Test text: '{text}'")
    print(f"Predicted: '{prediction}' | Expected: '{y_test[index]}' - {correctness}\n")

# Calculate the accuracy
accuracy = correct_predictions / len(X_test)

print(f"Accuracy: {accuracy * 100:.2f}%")

# If you want to classify new instances interactively, you can do something like this:
while True:
    new_input = input("Enter a piece of text to classify or 'exit' to quit: ")
    if new_input.lower() == 'exit':
        break
    prediction = clf.predict([new_input])
    print(f"The text is classified as: {prediction[0]}")