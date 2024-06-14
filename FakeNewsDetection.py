import tkinter as tk
from tkinter import scrolledtext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
import pandas as pd
import re
from PIL import Image, ImageTk

# Load a sample of the data
true = pd.read_csv("TRUE.csv", encoding="latin1", nrows=100)
fake = pd.read_csv("FAKE.csv", encoding="latin1", nrows=100)

# Data preprocessing
true['label'] = 1
fake['label'] = 0
news = pd.concat([fake, true], axis=0)
news = news.drop(['Image', 'Web', 'Category', 'Date'], axis=1)
news.reset_index(drop=True, inplace=True)

# Text cleaning function
def wordopt(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d', '', text)
    text = re.sub(r'\n', ' ', text)
    return text

news['Statement'] = news['Statement'].apply(wordopt)

# Preparing the data
x = news['Statement']
y = news['label']

# Vectorization
vectorization = TfidfVectorizer()
x = vectorization.fit_transform(x)

# Model initialization and training
logistic_model = LogisticRegression()
decision_tree_model = DecisionTreeClassifier()
random_forest_model = RandomForestClassifier()
gradient_boosting_model = GradientBoostingClassifier()

logistic_model.fit(x, y)
decision_tree_model.fit(x, y)
random_forest_model.fit(x, y)
gradient_boosting_model.fit(x, y)

# Tkinter UI
root = tk.Tk()
root.title("Fake News Detection")

# Background Image
bg_image = Image.open("newsbgg.jpg")
bg_photo = ImageTk.PhotoImage(bg_image)
bg_label = tk.Label(root, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

# Title Label
title_label = tk.Label(root, text="JustNews", font=("Helvetica", 25), bg="white", fg="black")
title_label.place(relx=0.5, rely=0.1, anchor="center")

def classify_news():
    news_article = text_input.get("1.0",'end-1c')
    
    if not news_article.strip():
        result_label.config(text="Please enter some text.")
        return

    cleaned_news = wordopt(news_article)
    news_vector = vectorization.transform([cleaned_news])
    
    # Logistic Regression
    logistic_prediction = logistic_model.predict(news_vector)
    logistic_accuracy = logistic_model.score(x, y)
    logistic_report = classification_report(y, logistic_model.predict(x))
    print("Logistic Regression Report:", logistic_report)  # Add this line
    
    # Decision Tree
    decision_tree_prediction = decision_tree_model.predict(news_vector)
    decision_tree_accuracy = decision_tree_model.score(x, y)
    decision_tree_report = classification_report(y, decision_tree_model.predict(x))
    print("Decision Tree Report:", decision_tree_report)  # Add this line
    
    # Random Forest
    random_forest_prediction = random_forest_model.predict(news_vector)
    random_forest_accuracy = random_forest_model.score(x, y)
    random_forest_report = classification_report(y, random_forest_model.predict(x))
    print("Random Forest Report:", random_forest_report)  # Add this line
    
    # Gradient Boosting
    gradient_boosting_prediction = gradient_boosting_model.predict(news_vector)
    gradient_boosting_accuracy = gradient_boosting_model.score(x, y)
    gradient_boosting_report = classification_report(y, gradient_boosting_model.predict(x))
    print("Gradient Boosting Report:", gradient_boosting_report)  # Add this line
    
    result_label.config(text=f"Logistic Regression: Accuracy - {logistic_accuracy}, Prediction - {'Genuine' if logistic_prediction == 1 else 'Fake'}\n"
                             f"Decision Tree: Accuracy - {decision_tree_accuracy}, Prediction - {'Genuine' if decision_tree_prediction == 1 else 'Fake'}\n"
                             f"Random Forest: Accuracy - {random_forest_accuracy}, Prediction - {'Genuine' if random_forest_prediction == 1 else 'Fake'}\n"
                             f"Gradient Boosting: Accuracy - {gradient_boosting_accuracy}, Prediction - {'Genuine' if gradient_boosting_prediction == 1 else 'Fake'}\n"
                             f"\n"
                             f"Please click on the classifier name to view the detailed classification report.\n")

    # Open new windows with classification reports
    result_label.bind("<Button-1>", lambda event: open_classifier_window("Logistic Regression", logistic_report, "blue"))
    result_label.bind("<Button-2>", lambda event: open_classifier_window("Decision Tree", decision_tree_report, "green"))
    result_label.bind("<Button-3>", lambda event: open_classifier_window("Random Forest", random_forest_report, "orange"))
    result_label.bind("<Button-4>", lambda event: open_classifier_window("Gradient Boosting", gradient_boosting_report, "purple"))

    result_label.place(relx=0.5, rely=0.7, anchor="center")
    # Text Input
text_input = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=10)
text_input.place(relx=0.5, rely=0.3, anchor="center")

# Classification Button
classify_button = tk.Button(root, text="Classify", command=classify_news, bg="blue", fg="white")
classify_button.place(relx=0.5, rely=0.5, anchor="center")

# Result Label
result_label = tk.Label(root, text="", font=("Helvetica", 12), bg="black", fg="white", justify=tk.LEFT)
result_label.place(relx=0.5, rely=0.7, anchor="center")

def open_classifier_window(classifier_name, report, color):
    classifier_window = tk.Toplevel(root)
    classifier_window.title(f"{classifier_name} Classification Report")

    report_text = scrolledtext.ScrolledText(classifier_window, wrap=tk.WORD, width=80, height=20)
    report_text.tag_configure('center', justify='center')
    report_text.tag_configure('color', foreground=color, font=('Arial', 12, 'bold'))

    # Adding variations to the report content
    if classifier_name == "Logistic Regression":
        report_text_content = f"{report}\n\nThis is the Logistic Regression report."
    elif classifier_name == "Decision Tree":
        report_text_content = f"{report}\n\nThis is the Decision Tree report."
    elif classifier_name == "Random Forest":
        report_text_content = f"{report}\n\nThis is the Random Forest report."
    elif classifier_name == "Gradient Boosting":
        report_text_content = f"{report}\n\nThis is the Gradient Boosting report."

    report_text.insert(tk.END, report_text_content, 'color')
    report_text.tag_add('center', '1.0', 'end')
    report_text.pack()

root.mainloop()
