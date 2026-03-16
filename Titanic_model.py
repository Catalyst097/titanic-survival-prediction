import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# --- DATA LOADING ---
# Load the built-in titanic dataset from the Seaborn library
df = sns.load_dataset("titanic")

# Force pandas to show all columns when printing so nothing is hidden with "..."
pd.set_option('display.max_columns', len(df.columns))

# --- DATA CLEANING & ENCODING ---
# Convert text categories into numbers. 'drop_first=True' prevents redundant columns
# (e.g., instead of having 'sex_male' and 'sex_female', it just keeps 'sex_male')
df = pd.get_dummies(df, columns=["sex", "embarked", "embark_town", "class", "alive", "who", "adult_male", "alone"], dtype=int, drop_first=True)

# Remove 'deck' (too many missing values) and 'alive_yes' (it's a duplicate of our answer 'survived')
df = df.drop(columns=["deck", "alive_yes"])

# Fill missing age values with the average age so the model doesn't crash on 'NaN' values
df["age"] = df["age"].fillna(df["age"].mean())

# Convert all data (like floats) to integers to simplify the math for the model
df = df.astype(int)

# Verify that there are zero missing values remaining in the dataset
print(df.isna().sum())

# --- DATA SPLITTING ---
# Define 'x' as our features (the clues) by dropping the target column
x = df.drop(columns=["survived"])

# Define 'y' as our target (the answer we want to predict)
y = df["survived"]

# Split the data: 70% to teach the model (Train) and 30% to test its memory (Test)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# --- MODEL 1: LOGISTIC REGRESSION ---
# Initialize the model. 'max_iter=1000' gives the solver more time to find the best fit
model = LogisticRegression(max_iter=1000)

# The 'Fit' step: The model learns the relationship between x (features) and y (survived)
model.fit(X_train, y_train)

# Use the trained model to guess survival for the unseen test data
y_pred = model.predict(X_test)

# --- MODEL 2: RANDOM FOREST ---
from sklearn.ensemble import RandomForestClassifier

# Initialize a Forest with 100 individual Decision Trees voting together
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the forest using the same training data
rf_model.fit(X_train, y_train)

# Get predictions from the Random Forest
rf_pred = rf_model.predict(X_test)

# Print the final performance comparison
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_pred):.2%}")

# --- VISUALIZATION ---
# Extract which features had the most 'weight' in the Random Forest's decisions
importances = pd.Series(rf_model.feature_importances_, index=x.columns)

# Graph the top 5 most influential columns to see what mattered most (Age, Fare, etc.)
importances.nlargest(5).plot(kind='barh')
plt.title("Top 5 Factors for Survival")
plt.show()