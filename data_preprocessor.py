# import all necessary libraries here
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# The following codes read the messy_data.csv file, followed by the informationstructure and description of the dataset: 
messy_dataset = pd.read_csv('C:/Users/Zahin/Desktop/Semester 2/Machine Learning and AI/Assignment 1/Assignment1/Scripts/messy_data.csv')
messy_dataset.head()    
messy_dataset.info()
#messy_dataset.describe()
print(type(messy_dataset)) # The uploaded csv file, using Pandas, is already a DataFrame.
    
print("===================================================================================")
    
# 1. Impute Missing Values
def impute_missing_values(data, strategy='mean'): 
    """
    Fill missing values in the dataset.
    :param data: pandas DataFrame
    :param strategy: str, imputation method ('mean', 'median', 'mode')
    :return: pandas DataFrame
    """
    data_imputed_cols = []
    numerical_cols = data.select_dtypes(include = ['float64'])
    # numerical_cols = numbers.drop(columns = 'target') # Did not run this code. 

    for col in numerical_cols:
        
        if strategy == "mean": 
            a = numerical_cols[col].fillna(numerical_cols[col].mean()) # Pandas' .mean(), .median(), and .mode() were used.
            data_imputed_cols.append(a) 
        elif strategy == "median": 
            b = numerical_cols[col].fillna(numerical_cols[col].median())
            data_imputed_cols.append(b)
        elif strategy == "mode":
            c = numerical_cols[col].fillna(numerical_cols[col].mode()[0])
            data_imputed_cols.append(c)
        else:
            data_imputed_cols.append("ERROR")
            
    updated_cols = pd.DataFrame(data_imputed_cols)
    
    return(pd.DataFrame(updated_cols.transpose())) # Initially, rows and columns were interchanged; I changed them to the original format (column names on top, row numbers on the left). 
    
imputed_data = impute_missing_values(messy_dataset) # This calls the function for imputing the missing values.


# 2. Remove Duplicates
def remove_duplicates(data):
    """
    Remove duplicate rows from the dataset.
    :param data: pandas DataFrame
    :return: pandas DataFrame
    """ 
    
    return(pd.DataFrame(data.drop_duplicates()))

no_duplicates = remove_duplicates(imputed_data)

# 3. Normalize Numerical Data
def normalize_data(data,method="minmax"):
    """
    Apply normalization to numerical features.
    :param data: pandas DataFrame
    :param method: str, normalization method ('minmax' (default) or 'standard')
    """
        
    if method == "minmax":
        scalar = MinMaxScaler()
        normalized_data = scalar.fit_transform(data)
        print("The Min-Max method was applied to the messy dataset.")
    elif method == "standard":
        scalar = StandardScaler()
        normalized_data = scalar.fit_transform(data)
        print("The Standardization method was applied to the messy dataset.")
    elif method == "log2": 
        normalized_data = scalar.fit_transform(np.log2(data + 1))  #log2 was implemented to reduce the skewness observed in the provided dataset. +1 was added because log(0) = undefined.
        print("The logarithmic method was applied to the messy dataset.")
    else:
        print("ERROR")
        
    return(pd.DataFrame(normalized_data))

normalized = normalize_data(no_duplicates) 
print(normalized)

# 4. Remove Redundant Features   
def remove_redundant_features(data, threshold=0.9):
    """  Remove redundant or duplicate columns.
    :param data: pandas DataFrame
    :param threshold: float, correlation threshold
    :return: pandas DataFrame
      """
    
    correlation = data.corr()
    print(correlation)
    
    absolute = correlation.abs()

    
    for i in range(0, len(absolute.columns)):
        
        for element in absolute[[i]]:
            
            if element < threshold:
                
               no_redundancy = absolute.drop(absolute.columns[[i]], axis = 1)

    return(pd.DataFrame(no_redundancy))

finalData = remove_redundant_features(normalized)
print(finalData)
# ---------------------------------------------------

def simple_model(input_data, split_data=True, scale_data=False, print_report=False):
    """
    A simple logistic regression model for target classification.
    Parameters:
    input_data (pd.DataFrame): The input data containing features and the target variable 'target' (assume 'target' is the first column).
    split_data (bool): Whether to split the data into training and testing sets. Default is True.
    scale_data (bool): Whether to scale the features using StandardScaler. Default is False.
    print_report (bool): Whether to print the classification report. Default is False.
    Returns:
    None
    The function performs the following steps:
    1. Removes columns with missing data.
    2. Splits the input data into features and target.
    3. Encodes categorical features using one-hot encoding.
    4. Splits the data into training and testing sets (if split_data is True).
    5. Scales the features using StandardScaler (if scale_data is True).
    6. Instantiates and fits a logistic regression model.
    7. Makes predictions on the test set.
    8. Evaluates the model using accuracy score and classification report.
    9. Prints the accuracy and classification report (if print_report is True).
    """

    #if there's any missing data, remove the columns
    input_data.dropna(inplace=True)

    # split the data into features and target
    target = input_data.copy()[input_data.columns[0]]
    features = input_data.copy()[input_data.columns[1:]]
    

    # if the column is not numeric, encode it (one-hot)
    for col in features.columns:
        if features[col].dtype == 'object':
            features = pd.concat([features, pd.get_dummies(features[col], prefix=col)], axis=1)
            features.drop(col, axis=1, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, stratify=target, random_state=42)

    if scale_data:
        # scale the data
        X_train = normalize_data(X_train)
        X_test = normalize_data(X_test)
        
    # instantiate and fit the model
    log_reg = LogisticRegression(random_state=42, max_iter=100, solver='liblinear', penalty='l2', C=1.0)
    log_reg.fit(X_train, y_train)

    # make predictions and evaluate the model
    y_pred = log_reg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy}')
    
    # if specified, print the classification report
    if print_report:
        print('Classification Report:')
        print(report)
        print('Read more about the classification report: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html and https://www.nb-data.com/p/breaking-down-the-classification')
    
     


simple_model(finalData)