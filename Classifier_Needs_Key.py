from skllm.config import SKLLMConfig

SKLLMConfig.set_openai_key("USER KEY CAN BE FOUND ON OPENAI")
SKLLMConfig.set_openai_org("USER ORG CAN BE FOUND ON OPENAI")

import numpy as np
from skllm import FewShotGPTClassifier

# Define your training data. Replace these with your actual data.
X_train = [
    "Training Data goes HERE"
]

y_train = [
    "Secondary Training Data goes HERE"
]

# Initialize and train the classifier
clf = FewShotGPTClassifier(openai_model="gpt-3.5-turbo")  # Replace with your actual model name
clf.fit(X_train, y_train)

# Define your test data (new instances that are unseen during training)
X_test = [
    # Examples related to topic
    "Examples of TRUE data go here",
    
    # Examples NOT related to topic
    "Examples of FALSE data go here"
]

y_test = [
    "true",
    "true",
    "false",
    "false"
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
