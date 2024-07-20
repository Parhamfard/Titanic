#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

# Load the Titanic dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
titanic = pd.read_csv(url)

# Display the first five rows
print(titanic.head())

# Fill missing values in the 'Age' column with the median age
titanic['Age'].fillna(titanic['Age'].median(), inplace=True)

# Create a new column 'FamilySize' as the sum of 'SibSp' and 'Parch'
titanic['FamilySize'] = titanic['SibSp'] + titanic['Parch']

# Drop the 'Cabin' column
titanic.drop(columns=['Cabin'], inplace=True)

# Display the first five rows after manipulation
print(titanic.head())


# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns

# Calculate the survival rate
survival_rate = titanic['Survived'].mean()
print(f'Survival Rate: {survival_rate:.2f}')

# Plot the distribution of passengers' ages
plt.figure(figsize=(10, 6))
sns.histplot(titanic['Age'], bins=30, kde=True)
plt.title('Distribution of Passengers\' Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Create a bar plot showing the survival rate by passenger class
plt.figure(figsize=(10, 6))
sns.barplot(x='Pclass', y='Survived', data=titanic)
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.show()


# In[5]:



from scipy.stats import ttest_ind

# Separate survivors and non-survivors
survivors = titanic[titanic['Survived'] == 1]['Age']
non_survivors = titanic[titanic['Survived'] == 0]['Age']

# Perform a t-test
t_stat, p_value = ttest_ind(survivors, non_survivors)
print(f'T-statistic: {t_stat:.2f}, P-value: {p_value:.2f}')


# In[6]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Define the feature columns and target
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize']
target = 'Survived'

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(titanic[features], titanic[target], test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict on the testing set
y_pred = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')


# In[7]:


# Example dataset with missing and inconsistent data
data = {
    'Name': ['Alice', 'Bob', 'Charlie', None, 'Eve'],
    'Age': [25, 30, None, 22, 29],
    'Gender': ['F', 'M', 'M', 'F', None],
    'Salary': ['50000', '60000', '70000', '40000', '50000']
}
df = pd.DataFrame(data)

# Handle missing values
df['Name'].fillna('Unknown', inplace=True)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Gender'].fillna('Unknown', inplace=True)

# Correct data types
df['Salary'] = df['Salary'].astype(int)

# Standardize categorical variables
df['Gender'] = df['Gender'].map({'F': 'Female', 'M': 'Male', 'Unknown': 'Unknown'})

print(df)


# In[ ]:





# In[ ]:




