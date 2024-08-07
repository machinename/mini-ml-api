from flask import jsonify
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


## To-Do Add Settings

def regression(df, label: str):
    try:    
        X = df.drop(columns=[label])
        y = df[label]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
        model = LinearRegression()
            
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        evaluation_metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'rmse': mean_squared_error(y_test, y_pred, squared=False), 
        }
    
        return model, evaluation_metrics, None, None
    except Exception as e:
        print("Error:", e)
        return None, None, jsonify({'Error': 'Error Occured While Training Model'}), 500

def logistic_classification(df, label: str):
    try:
        X = df.drop(columns=[label])
        y = df[label]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
        model = LogisticRegression()
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        evaluation_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
        }
    
        return model, evaluation_metrics, None, None
    except Exception as e:
        print("Error:", e)
        return None, None, jsonify({'Error': 'Error Occured While Training Model'}), 500