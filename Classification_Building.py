# Importing necessary libraries
from skllm import FewShotGPTClassifier
from sklearn.model_selection import train_test_split

# Step 1: Data Preparation
# Assuming you've loaded your data into two lists, `texts` and `labels`
# `texts` contains the advertisements (or text snippets)
# `labels` contains binary labels: 1 for trafficking ads and 0 for non-trafficking ads

texts = ["Sample ad 1", "Sample ad 2", ...]  # Your dataset here
labels = [0, 1, ...]  # Corresponding labels for your dataset

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Step 2: Training the Few-Shot Classifier
clf = FewShotGPTClassifier(openai_model="gpt-3.5-turbo")
clf.fit(X_train, y_train)

# Step 3: Classifying user prompts
while True:
    user_input = input("Enter the advertisement text to analyze or 'exit' to quit: ")

    if user_input.lower() == 'exit':
        break

    prediction = clf.predict([user_input])

    if prediction[0] == 1:
        print("This advertisement shows signs of human trafficking.")
    else:
        print("This advertisement doesn't show clear signs of human trafficking.")
