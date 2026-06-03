import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

def load_vendor_invoice_data(db_path: str):
    """
    Load vendor invoice data from SQLite database.
    """
    
    con = sqlite3.connect(db_path) 
    query = "SELECT * FROM vendor_invoice"
    
    
    df = pd.read_sql_query(query, con)
    con.close()
    return df



def prepare_features(df: pd.DataFrame):
    """
    SELECT features and target variable.
    """

    X = df[["Dollars"]]
    Y = df["Freight"]
    return X,Y

def split_data(X,Y,test_size=0.2,random_state=42):
    """
    Split dataset into train and test splits
    """
    
    return train_test_split(X,Y,test_size=test_size,random_state=random_state)