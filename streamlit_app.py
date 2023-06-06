import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore') 

df = pd.read_excel('bank-additional.xlsx')
df.rename(columns={'y': 'bank term deposit'}, inplace=True)
df = df[df['housing'] != 'unknown']
df.head()

housing_dict = {'no': 0, 'yes': 1}
default_value = -1
df['housing'] = df['housing'].apply(lambda x: housing_dict.get(x, default_value))
df.head()
df = df[df['marital'] != 'unknown']
df.head()
marital_dict = {'married': 0, 'single': 1}
default_value = -1

df['marital'] = df['marital'].apply(lambda x: marital_dict.get(x, default_value))
df.head()
df = df[
    (df['job'] != 'unknown') &
    (df['education'] != 'unknown') &
    (df['contact'] != 'unknown') &
    (df['month'] != 'unknown') &
    (df['day_of_week'] != 'unknown') &
    (df['poutcome'] != 'unknown')
]

df.head()
cols = ['job','education','contact','month','day_of_week','poutcome']
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in cols:
    df[col] = le.fit_transform(df[col])
df.head()
df = df[df['default'] != 'unknown']
df.head()
default_dict = {'no': 0, 'yes': 1}
default_value = -1

df['default'] = df['default'].apply(lambda x: default_dict.get(x, default_value))
df.head()
df = df[df['loan'] != 'unknown']
df.head()
loan_dict = {'no': 0, 'yes': 1}
default_value = -1

df['loan'] = df['loan'].apply(lambda x: loan_dict.get(x, default_value))
df.head()
df = df[df['bank term deposit'] != 'unknown']
df.head()
y_dict = {'no':0, 'yes':1}
df['bank term deposit'] = df['bank term deposit'].apply(lambda x: y_dict.get(x, default_value))
df.head()
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
Upper_bound = Q3 + 1.5 * IQR
outlier_rows = ((df < lower_bound) | (df > Upper_bound)).any(axis=1)
df_no_outliers = df[~outlier_rows]
num_outliers = sum(outlier_rows)
print(f"Number of outliers removed: {num_outliers}")
df.head()
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
X
y = df['bank term deposit']
y
X = df.drop(columns=['bank term deposit','nr.employed',"cons.price.idx","cons.conf.idx", "euribor3m","emp.var.rate","pdays","contact"])
X
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import accuracy_score

# Generate a synthetic classification dataset
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the ElasticNetCV model
elastic_net_cv = ElasticNetCV(cv=5, random_state=42)

# Fit ElasticNetCV to the training data
elastic_net_cv.fit(X_train, y_train)

# Retrieve the best alpha and l1_ratio values
best_alpha = elastic_net_cv.alpha_
best_l1_ratio = elastic_net_cv.l1_ratio_

# Create the Logistic Regression classifier with the best hyperparameters
clf = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=best_l1_ratio, C=1/best_alpha)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Decision Tree Classifier
clf = DecisionTreeClassifier()

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score

# Generate a synthetic classification dataset
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Ridge classifier model
clf = RidgeClassifier()

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score

# Generate a synthetic classification dataset
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Extra Trees Classifier
clf = ExtraTreesClassifier()

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Generate a synthetic classification dataset
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Random Forest Classifier
clf = RandomForestClassifier()

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Generate a synthetic classification dataset
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Gradient Boosting Classifier
clf = GradientBoostingClassifier()

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
xgb_model = xgb.XGBClassifier(
    learning_rate=0.1,
    n_estimators=100,
    max_depth=3,
    objective='binary:logistic',
    random_state=42
)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

import sys
pip install xgboost
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Apply feature scaling
    ('xgb_model', xgb.XGBClassifier(
        learning_rate=0.1,
        n_estimators=100,
        max_depth=3,
        objective='binary:logistic',
        random_state=42
    ))
])

# Train the pipeline on the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Apply feature scaling
    ('grid_search', GridSearchCV(
        GradientBoostingClassifier(),
        param_grid={
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.1, 0.05, 0.01],
            'max_depth': [3, 4, 5]
        },
        cv=5
    ))
])

# Train the pipeline on the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Best Parameters:", pipeline.named_steps['grid_search'].best_params_)

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Generate a synthetic classification dataset
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the steps of the pipeline
steps = [
    ('scaler', StandardScaler()),  # Apply feature scaling
    ('clf', RandomForestClassifier())  # Random Forest Classifier
]

# Create the pipeline
pipeline = Pipeline(steps)

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from decisiontree import DecisionTree

# Define a function to make predictions
def predict_term(data):
    # Load the saved model
    with open('chaid_model.pkl', 'rb') as f:
        model = pickle.load(f)
    prediction = model.predict(data)
    print(prediction)
    if prediction[0] == 0:
        return "No"
    else:
        return "Yes"
      def main():
    # Set the app title
    st.set_page_config(page_title='Subscription to time deposits')

    # Set the app heading
    st.title('Subscription to time deposits')
# Define the input fields for the app
    age = st.text_input("Age[18-100]", 18,100)
    job = st.selectbox("Job", ["Admin.", "Blue-collar", "Entrepreneur", "Housemaid", "Management", "Retired", "Self-employed", "Services", "Student", "Technician", "Unemployed", "Unknown"])
    marital = st.selectbox("Marital Status", ["Divorced", "Married", "Single", "Unknown"])
    education = st.selectbox("Education Level", ["Basic 4y", "Basic 6y", "Basic 9y", "High School", "Professional Course", "University Degree", "Unknown", "Illiterate"])
    default = st.radio("Has Credit in Default?", ("No", "Yes", "Unknown"))
    housing = st.radio("Has Housing Loan?", ("No", "Yes", "Unknown"))
    loan = st.radio("Has Personal Loan?", ("No", "Yes", "Unknown"))
    contact = st.selectbox("Contact Communication Type", ["Cellular", "Telephone"])
    month = st.selectbox("Last Contact Month", ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    day_of_week = st.selectbox("Last Contact Day of the Week", ["Mon", "Tue", "Wed", "Thu", "Fri"])
    duration = st.text_input("Last Contact Duration (Seconds)[0-5000]", 0,5000)
    campaign = st.slider("Number of Contacts During Campaign", 1,50,1)
    previous = st.slider("Number of Contacts Before This Campaign", 0,10,1)
    poutcome = st.selectbox("Outcome of Previous Marketing Campaign", ["Failure", "Nonexistent", "Success"])
    
    data = {"age": age,
            "job": job,
            "marital": marital,
            "education": education,
            "default": default,
            "housing": housing,
            'loan': loan,
            'contact': contact,
            'month': month,
            'day_of_week': day_of_week,
            'duration': duration,
            'campaign': campaign,
            'previous': previous,
            'poutcome': poutcome,
           }
    # Convert the dictionary into a dataframe
    data_df = pd.DataFrame(data, index=[0])

# Apply one-hot encoding to the categorical columns
    cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan',
    'contact', 'month', 'day_of_week', 'poutcome']

    data_encoded = pd.get_dummies(data_df, columns=cat_cols)

    prediction=""
    if st.button("Predict"):
        prediction=predict_term(data_encoded)
        st.success(f"The predicted status is: {prediction}")
    

if __name__=='__main__':
    main()
      
