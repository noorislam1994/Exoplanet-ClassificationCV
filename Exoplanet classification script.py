#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import packages
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#Preprocessing libraries
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
# Sklearn packages for evaluation, modeling and preformance
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import mean_squared_error, precision_score, confusion_matrix, accuracy_score
#Keras packages relating to Deep Learning Architecture
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
# Visualizes all the columns
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# In[2]:


pip install imblearn --user


# In[3]:


pip install keras


# In[4]:


pip install tensorflow


# In[57]:


#Read the kepler dataset
exop = pd.read_csv('keplerm.csv')

# Select top of the dataset
exop.head()


# In[58]:


#Checking the size of the data
num_rows, num_columns = exop.shape

print(f'Number of rows: {num_rows}')
print(f'Number of columns: {num_columns}')

#Checking for the datatype of the metro dataset

print(type(exop))


# In[61]:


new_headers_name = ["KepID", 
                    "KOI Name", 
                    "Kepler Name", 
                    "Exoplanet Archive Disposition", 
                    "Disposition Using Kepler Data", 
                    "Disposition Score", 
                    "Not Transit-Like False Positive Flag",
                    "Stellar Eclipse False Positive Flag", 
                    "Centroid Offset False Positive Flag", 
                    "Ephemeris Match Indicates Contamination False Positive Flag", 
                    "Orbital Period in days", 
                    "Orbital Period Upper Unc.", 
                    "Orbital Period Lower Unc.", 
                    "Transit Epoch [BKJD]", 
                    "Transit Epoch Upper Unc. [BKJD]", 
                    "Transit Epoch Lower Unc. [BKJD]", 
                    "Impact Parameter", 
                    "Impact Parameter Upper Unc.", 
                    "Impact Parameter Lower Unc.", 
                    "Transit Duration [hrs]", 
                    "Transit Duration Upper Unc. [hrs]", 
                    "Transit Duration Lower Unc. [hrs]", 
                    "Transit Depth [ppm]", 
                    "Transit Depth Upper Unc. [ppm]", 
                    "Transit Depth Lower Unc. [ppm]", 
                    "Planetary Radius [Earth radii]",
                    "Planetary Radius Upper Unc. [Earth radii]", 
                    "Planetary Radius Lower Unc. [Earth radii]", 
                    "Equilibrium Temperature [K]", 
                    "Equilibrium Temperature Upper Unc. [K]", 
                    "Equilibrium Temperature Lower Unc. [K]",
                    "Insolation Flux [Earth flux]", 
                    "Insolation Flux Upper Unc. [Earth flux]", 
                    "Insolation Flux Lower Unc. [Earth flux]",
                    "Transit Signal-to-Noise", 
                    "TCE Planet Number", 
                    "TCE Delivery", 
                    "Stellar Effective Temperature [K]", 
                    "Stellar Effective Temperature Upper Unc. [K]", 
                    "Stellar Effective Temperature Lower Unc. [K]", 
                    "Stellar Surface Gravity [log10(cm/s**2)]", 
                    "Stellar Surface Gravity Upper Unc. [log10(cm/s**2)]", 
                    "Stellar Surface Gravity Lower Unc. [log10(cm/s**2)]", 
                    "Stellar Radius [Solar radii]", 
                    "Stellar Radius Upper Unc. [Solar radii]", 
                    "Stellar Radius Lower Unc. [Solar radii]", 
                    "RA [decimal degrees]", 
                    "Dec [decimal degrees]", 
                    "Kepler-band [mag]"]

# Assign the new header names to the DataFrame
exop.columns = new_headers_name


# In[62]:


exop.head()


# In[63]:


# Check the data types of the columns using d.types attribute

exop.info()


# In[10]:


# Identify missing values
missing = exop.isnull()

# Count the number of missing values in each column
missing_count = missing.sum()

# Display the missing values and their count
print(missing_count)

# Calculate the percentage of rows with missing values in the dataset
percentage = exop.isnull().any(axis=1).mean().round(4) * 100

# Print the percentage of rows with missing values
print(percentage)


# In[11]:


# Calculate the percentage of missing values per column
percentage1 = exop.isna().mean() * 100

# Print the percentage of missing values per column
print(percentage1)


# In[12]:


exop['Candidate ExopStatus'] = exop['Disposition Using Kepler Data'].apply(lambda x: 1 if x == 'CANDIDATE' else 0)


# In[13]:


#We will be dropping null-columns and irrelevant columns

columns_to_drop_1 = ['KepID', 'KOI Name', 'Kepler Name', 
                     'Disposition Score', 'Equilibrium Temperature Lower Unc. [K]', 
                     'Equilibrium Temperature Upper Unc. [K]', 'TCE Delivery', 'Disposition Using Kepler Data']
exop = exop.drop(columns_to_drop_1, axis=1)

#Lastly, we will drop the 'Exoplanet Archive Disposition' since we're interested in 'Disposition Using Kepler Data'

exop = exop.drop(['Exoplanet Archive Disposition'], axis=1)


# In[14]:


#Convert object columns into integer datatypes
exop['Not Transit-Like False Positive Flag'] = exop['Not Transit-Like False Positive Flag'].astype(int)
exop['Stellar Eclipse False Positive Flag'] = exop['Stellar Eclipse False Positive Flag'].astype(int)
exop['Centroid Offset False Positive Flag'] = exop['Centroid Offset False Positive Flag'].astype(int)
exop['Ephemeris Match Indicates Contamination False Positive Flag'] = exop['Ephemeris Match Indicates Contamination False Positive Flag'].astype(int)


# In[15]:


#Drop missing values
exop.dropna(inplace=True)
# Display the number of rows and columns after dropping the missing values
print(f'After dropping missing values: {exop.shape[0]} rows, {exop.shape[1]} columns')


# In[16]:


# Check the data types of the columns using d.types attribute after cleansing

exop.info()


# In[17]:


exop


# In[18]:


#Checking for duplicates in the dataset

duplicates = exop[exop.duplicated()]

#Count for duplicates in the dataset

num_duplicates = duplicates.count()

print(num_duplicates)


# In[19]:


# Calculate descriptive statistics for the dataset
descriptive_stats = exop.describe().style

# Set title for the table
descriptive_stats.set_caption("Table 01: Descriptive Statistics for the Metro Dataset")

# Display the table
display(descriptive_stats)


# In[20]:


#Selecting numerical columns
columns = ["Orbital Period in days", 
                    "Orbital Period Upper Unc.", 
                    "Orbital Period Lower Unc.", 
                    "Transit Epoch [BKJD]", 
                    "Transit Epoch Upper Unc. [BKJD]", 
                    "Transit Epoch Lower Unc. [BKJD]", 
                    "Impact Parameter", 
                    "Impact Parameter Upper Unc.", 
                    "Impact Parameter Lower Unc.", 
                    "Transit Duration [hrs]", 
                    "Transit Duration Upper Unc. [hrs]", 
                    "Transit Duration Lower Unc. [hrs]", 
                    "Transit Depth [ppm]", 
                    "Transit Depth Upper Unc. [ppm]", 
                    "Transit Depth Lower Unc. [ppm]", 
                    "Planetary Radius [Earth radii]",
                    "Planetary Radius Upper Unc. [Earth radii]", 
                    "Planetary Radius Lower Unc. [Earth radii]", 
                    "Equilibrium Temperature [K]", 
                    "Insolation Flux [Earth flux]", 
                    "Insolation Flux Upper Unc. [Earth flux]", 
                    "Insolation Flux Lower Unc. [Earth flux]",
                    "Transit Signal-to-Noise", 
                    "TCE Planet Number", 
                    "Stellar Effective Temperature [K]", 
                    "Stellar Effective Temperature Upper Unc. [K]", 
                    "Stellar Effective Temperature Lower Unc. [K]", 
                    "Stellar Surface Gravity [log10(cm/s**2)]", 
                    "Stellar Surface Gravity Upper Unc. [log10(cm/s**2)]", 
                    "Stellar Surface Gravity Lower Unc. [log10(cm/s**2)]", 
                    "Stellar Radius [Solar radii]", 
                    "Stellar Radius Upper Unc. [Solar radii]", 
                    "Stellar Radius Lower Unc. [Solar radii]", 
                    "RA [decimal degrees]", 
                    "Dec [decimal degrees]", 
                    "Kepler-band [mag]"]

num_plots = len(columns)
rows = (num_plots // 2) + (num_plots % 2)
fig, axs = plt.subplots(rows, 2, figsize=(20, 6*rows))

for i, col in enumerate(columns):
    sns.distplot(exop[col], ax=axs[i//2, i%2])
    axs[i//2, i%2].set_title('Density Distribution of {}'.format(col))

# remove any unused subplots
if num_plots < rows*2:
    for ax in axs[-1]:
        ax.remove()
        
plt.tight_layout()
plt.show()


# In[21]:


# Count the number of instances for each class label
class_counts = exop['Candidate ExopStatus'].value_counts()

# Create a pie chart
plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%')
plt.title('Class Label Balance')
plt.show()


# In[22]:


# Calculate the IQR for each column
q1 = exop[columns].quantile(0.25)
q3 = exop[columns].quantile(0.75)
iqr = q3 - q1

# Define the outlier threshold for each column
outlier_threshold = 1.5 * iqr

# Find the outliers for each column
outliers = ((exop[columns] < (q1 - outlier_threshold)) | (exop[columns] > (q3 + outlier_threshold)))

# Count the number of outliers for each column
num_outliers = outliers.sum()

# Print the number of outliers per column
print(num_outliers)


# In[23]:


# Calculate the IQR for each column
q1 = exop[columns].quantile(0.25)
q3 = exop[columns].quantile(0.75)
iqr = q3 - q1

# Define the outlier threshold for each column
outlier_threshold = 1.5 * iqr

# Find the outliers for each column
outliers = ((exop[columns] < (q1 - outlier_threshold)) | (exop[columns] > (q3 + outlier_threshold)))

# Count the number of outliers for each column
num_outliers = outliers.sum()

# Calculate the percentage of outliers for each column
percent_outliers = (num_outliers / len(exop)) * 100

# Print the percentage of outliers per column
print("Percentage of outliers per column:")
print(percent_outliers)


# In[24]:


# create a copy of the dataset to apply outlier removal
exop_copy = exop.copy()

# find the outliers for each column
outliers = {}
for column in columns:
    outliers[column] = exop_copy[(exop_copy[column] < (q1[column] - outlier_threshold[column])) | 
                                 (exop_copy[column] > (q3[column] + outlier_threshold[column]))][column].index.tolist()

# combine all outliers into a single list
all_outliers = set(outliers[columns[0]])
for column in columns[1:]:
    all_outliers = all_outliers.union(set(outliers[column]))

# calculate the percentage of outliers
percentage_outliers = (len(all_outliers) / len(exop)) * 100

print(f"Percentage of total instances represented as outliers: {percentage_outliers:.2f}%")


# In[25]:


# Compute the 10th and 90th percentiles for each column
q10 = exop_copy.quantile(0.1)
q90 = exop_copy.quantile(0.9)

# Replace values below 10th percentile with 10th percentile value
# Replace values above 90th percentile with 90th percentile value
exop_copy = exop_copy.clip(lower=q10, upper=q90, axis=1)


# In[26]:


# Calculate the IQR for each column
q1 = exop_copy[columns].quantile(0.25)
q3 = exop_copy[columns].quantile(0.75)
iqr = q3 - q1

# Define the outlier threshold for each column
outlier_threshold = 1.5 * iqr

# Find the outliers for each column
outliers = ((exop_copy[columns] < (q1 - outlier_threshold)) | (exop_copy[columns] > (q3 + outlier_threshold)))

# Count the number of outliers for each column
num_outliers = outliers.sum()

# Calculate the percentage of outliers for each column
percent_outliers = (num_outliers / len(exop_copy)) * 100

# Print the percentage of outliers per column
print("Percentage of outliers per column:")
print(percent_outliers)


# In[27]:


# Create the correlation matrix
corr_matrix = exop_copy[columns].corr()

# Set the figure size
plt.figure(figsize=(25, 25))

# Plot the correlation matrix using a heatmap
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, fmt='.2f')
plt.show()


# In[28]:


# Select upper triangle of correlation matrix
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.7
high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column].abs() > 0.7)]

# Print the highly correlated features
print("Highly correlated features:")
for feature in high_corr_features:
    print(feature)


# In[29]:


# Get the absolute correlation values for all features with the class label

corr_matrix2 = exop_copy.corr()

class_corr = abs(corr_matrix2['Candidate ExopStatus'])

# Get the features highly correlated with the class label
high_corr_feats = class_corr[class_corr > 0.1].sort_values(ascending=False)
print("Highly correlated features with the class label:")
print(high_corr_feats)

# Get the least correlated features with the class label
least_corr_feats = class_corr[class_corr < 0.1].sort_values(ascending=True)
print("\nLeast correlated features with the class label:")
print(least_corr_feats)


# In[30]:


# define the feature and target variable
X = exop_copy.drop(['Candidate ExopStatus'], axis=1)
y = exop_copy['Candidate ExopStatus']


# In[31]:


# drop the highly correlated features from the X dataset
X = X.drop(high_corr_features, axis=1)


# In[73]:


print("Number of rows:", X.shape[0])
print("Number of columns:", X.shape[1])


# In[32]:


#Scaling the dataset

# make a copy of X and y
X_pr = X.copy()
y_pr = y.copy()

scaler = StandardScaler()
X_pr = scaler.fit_transform(X_pr)


# In[33]:


# create a PCA object with n_components=None
pca = PCA(n_components=None)

# fit the PCA model on the data
pca.fit(X_pr)

# plot the explained variance ratio for each component
plt.plot(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, marker='o')
plt.xlabel('Component')
plt.ylabel('Explained Variance Ratio')
plt.show()


# In[34]:


# instantiate PCA
pca = PCA(n_components=5)

# fit and transform PCA on X_pr
X_pr = pca.fit_transform(X_pr)


# In[35]:


#Resample the data to balance the class label

smote = SMOTE(random_state=42)
X_pr, y_pr = smote.fit_resample(X_pr, y_pr)


# In[36]:


# Count the number of instances for each class label
class_smote = y_pr.value_counts()

# Create a pie chart
plt.pie(class_smote, labels=class_smote.index, autopct='%1.1f%%')
plt.title('Class Label Balance')
plt.show()


# In[37]:


print(X_pr.shape)
print(y_pr.shape)


# In[38]:


# create k-fold object
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# create models
svm_model = SVC()
dt_model = DecisionTreeClassifier()
nn_model = MLPClassifier()

# create a list of models and their names
models = [("SVM", svm_model), ("Decision Tree", dt_model), ("Neural Network", nn_model)]

# iterate through the list of models and evaluate each model using cross-validation
for model_name, model in models:
    # compute cross-validation scores
    scores = cross_validate(model, X_pr, y_pr, cv=kf, scoring=["accuracy", "precision", "f1", "recall"])
    
    # print the results
    print(f"{model_name} Results:")
    print(f"Accuracy: {scores['test_accuracy'].mean():.3f} (+/- {scores['test_accuracy'].std():.3f})")
    print(f"Precision: {scores['test_precision'].mean():.3f} (+/- {scores['test_precision'].std():.3f})")
    print(f"F1-Score: {scores['test_f1'].mean():.3f} (+/- {scores['test_f1'].std():.3f})")
    print(f"Recall: {scores['test_recall'].mean():.3f} (+/- {scores['test_recall'].std():.3f})")


# In[74]:


# create a list of models and their names
models = [("SVM", svm_model), ("Decision Tree", dt_model), ("Neural Network", nn_model)]

# create a list to store the cross-validation scores for each model
scores_list = []

# iterate through the list of models and evaluate each model using cross-validation
for model_name, model in models:
    # compute cross-validation scores
    scores = cross_validate(model, X_pr, y_pr, cv=kf, scoring=["accuracy"])
    scores_list.append(scores['test_accuracy'])

# create a dataframe from the scores list
df = pd.DataFrame(scores_list, index=[model_name for model_name, model in models]).T

# plot a boxplot to visualize the accuracy of the models
fig, ax = plt.subplots()
df.plot(kind='box', ax=ax)
ax.set_title('Accuracy of Models')
ax.set_ylabel('Accuracy')
plt.show()


# In[40]:


#Define the hyperparameters and their ranges to search over
svm_param_grid = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"],
    "gamma": [0.01, 0.1, 1]
}
dt_param_grid = {
    "max_depth": [5, 10, 15],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 5]
}
nn_param_grid = {
    "hidden_layer_sizes": [(50,), (100,), (200,), (50,50), (100,50)],
    "activation": ["relu", "tanh"],
    "alpha": [0.0001, 0.001, 0.01]
}

#Define a list of models and their corresponding hyperparameter grids
models = [
    ("SVM", svm_model, svm_param_grid),
    ("Decision Tree", dt_model, dt_param_grid),
    ("Neural Network", nn_model, nn_param_grid)
]

# Iterate through the list of models and perform GridSearchCV
for model_name, model, param_grid in models:
    # Perform GridSearchCV
    grid_search = GridSearchCV(
    model, param_grid, cv=5, n_jobs=-1, scoring=["accuracy", "precision", "f1", "recall"], refit="accuracy"
)
    grid_search.fit(X_pr, y_pr)
    
    # Print the best hyperparameters and performance metrics
    print(f"{model_name} Results:")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Accuracy: {grid_search.best_score_:.3f}\n")


# In[48]:


#Define models with best hyperparameters obtained from GridSearch
svm_model_ = SVC(C=10, gamma=1, kernel="rbf")
dt_model_ = DecisionTreeClassifier(max_depth=15, min_samples_leaf=1, min_samples_split=2)
nn_model_ = MLPClassifier(activation="tanh", alpha=0.0001, hidden_layer_sizes=(100, 50))

# Create a list of models 
models = [("SVM", svm_model_), ("Decision Tree", dt_model_), ("Neural Network", nn_model_)]

# plot the learning curve for each model
for model_name, model in models:
    # compute the learning curve
    train_sizes, train_scores, test_scores = learning_curve(model, X_pr, y_pr, cv=10, n_jobs=-1, 
                                                            train_sizes=np.linspace(0.1, 1.0, 10),
                                                            scoring="accuracy")
    # calculate the mean and standard deviation of the training and test scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # plot the learning curve
    plt.figure(figsize=(8, 6))
    plt.title(f"{model_name} Learning Curve")
    plt.xlabel("Training Examples")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_mean, "o-", color="r", label="Training Score")
    plt.plot(train_sizes, test_mean, "o-", color="g", label="Cross-Validation Score")
    plt.legend(loc="best")
    plt.show()


# In[64]:


#Create model

# split the dataset into training and test sets using stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(X.shape,X_train.shape)


# In[65]:


#Scaling the dataset 

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[66]:


# Use PCA to reduce the dimensionality of the training data
pca = PCA(n_components=5)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


# In[67]:


# Apply SMOTE only on the training set
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)


# In[68]:


svm_model_op = SVC(C=100, gamma=1, kernel="rbf")
dt_model_op = DecisionTreeClassifier(max_depth=15, min_samples_leaf=1, min_samples_split=2)
nn_model_op = MLPClassifier(activation="tanh", alpha=0.0001, hidden_layer_sizes=(100, 50))


# fit the models to the data
svm_model_op.fit(X_train, y_train)
nn_model_op.fit(X_train, y_train)
dt_model_op.fit(X_train, y_train)

# make predictions on the test data
svm_pred = svm_model_op.predict(X_test)
nn_pred = nn_model_op.predict(X_test)
dt_pred = dt_model_op.predict(X_test)

# evaluate the model performance
svm_accuracy = accuracy_score(y_test, svm_pred)
svm_precision = precision_score(y_test, svm_pred)
svm_recall = recall_score(y_test, svm_pred)
svm_f1 = f1_score(y_test, svm_pred)

nn_accuracy = accuracy_score(y_test, nn_pred)
nn_precision = precision_score(y_test, nn_pred)
nn_recall = recall_score(y_test, nn_pred)
nn_f1 = f1_score(y_test, nn_pred)

dt_accuracy = accuracy_score(y_test, dt_pred)
dt_precision = precision_score(y_test, dt_pred)
dt_recall = recall_score(y_test, dt_pred)
dt_f1 = f1_score(y_test, dt_pred)

# print the evaluation metrics
print("SVM Evaluation Metrics:")
print("Accuracy: ", svm_accuracy)
print("Precision: ", svm_precision)
print("Recall: ", svm_recall)
print("F1-score: ", svm_f1)

print("Neural Network Evaluation Metrics:")
print("Accuracy: ", nn_accuracy)
print("Precision: ", nn_precision)
print("Recall: ", nn_recall)
print("F1-score: ", nn_f1)

print("Decision Tree Evaluation Metrics:")
print("Accuracy: ", dt_accuracy)
print("Precision: ", dt_precision)
print("Recall: ", dt_recall)
print("F1-score: ", dt_f1)

# calculate the confusion matrix for each model
svm_cm = confusion_matrix(y_test, svm_pred)
nn_cm = confusion_matrix(y_test, nn_pred)
dt_cm = confusion_matrix(y_test, dt_pred)

# print the confusion matrices
print("SVM Confusion Matrix:")
print(svm_cm)

print("Neural Network Confusion Matrix:")
print(nn_cm)

print("Decision Tree Confusion Matrix:")
print(dt_cm)


# In[69]:


#Setting the model hyperparameters 
model = Sequential()
model.add(Dense(units=100, input_dim=X_train.shape[1], activation='tanh'))
model.add(Dense(units=50, activation='tanh'))
model.add(Dense(units=1, activation='sigmoid'))

#Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model performance
nn_pred = model.predict(X_test)
nn_pred = np.argmax(nn_pred, axis=1)
nn_accuracy = accuracy_score(y_test, nn_pred)
nn_precision = precision_score(y_test, nn_pred)
nn_recall = recall_score(y_test, nn_pred)
nn_f1 = f1_score(y_test, nn_pred)

# print the evaluation metrics
print("Neural Network Evaluation Metrics:")
print("Accuracy: ", nn_accuracy)
print("Precision: ", nn_precision)
print("Recall: ", nn_recall)
print("F1-score: ", nn_f1)

# Calculate the confusion matrix
nn_cm = confusion_matrix(y_test, nn_pred)

# print the confusion matrix
print("Neural Network Confusion Matrix:")
print(nn_cm)


# In[70]:


#Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)

#print the test set loss and accuracy
print("Test set Loss: ", test_loss)
print("Test set Accuracy: ", test_accuracy)


# In[ ]:




