import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

# Step 1: Load Data from CSV
df = pd.read_csv("student_science_levels.csv")  # Change this to your actual CSV file path

# Step 2: Define Features and Target
X = df[["Age", "School_Level", "Science_Grade"]]  # Features: Age, School_Level, Science_Grade
y = df["Science_Level"]  # Target: Science_Level (Low, Intermediate, High)

# Step 3: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Define the Categories for the Categorical Feature (School_Level)
# Adjust this list based on your data's possible categories
all_categories = ['Primary', 'Secondary']  # Example categories

# Step 5: Create a Preprocessing Pipeline
# For numerical features, use StandardScaler
numeric_features = ["Age", "Science_Grade"]
numeric_transformer = StandardScaler()

# For categorical features, use OneHotEncoder
categorical_features = ["School_Level"]
categorical_transformer = OneHotEncoder(categories=[all_categories], handle_unknown='ignore')

# Create the full preprocessing pipeline using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Step 6: Create a Model Pipeline with RandomForestClassifier
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# Step 7: Train the Model
pipeline.fit(X_train, y_train)

# Step 8: Evaluate the Model
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Step 9: Save the Model and the Pipeline
with open("model_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Model training complete and pipeline saved!")
