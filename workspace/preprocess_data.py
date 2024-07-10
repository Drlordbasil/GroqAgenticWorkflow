def preprocess_data(data):
    # Handle missing values
    data.fillna(data.mean(), inplace=True)
    
    # Encode categorical variables
    data = pd.get_dummies(data, columns=['category'])
    
    # Scale numerical variables
    scaler = StandardScaler()
    data[['numerical_column']] = scaler.fit_transform(data[['numerical_column']])
    
    return data