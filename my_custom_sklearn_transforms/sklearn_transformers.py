from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')

class AvgFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, columns, new_col: str):
        self.columns = columns
        self.new_col = new_col

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):        
        data = X.copy()        
        data[self.new_col] = data[self.columns].values.mean(axis=1)

        return data

class BinFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, columns, new_col):
        self.columns = columns
        self.new_col = new_col

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        data = X.copy()
        condition = data[self.columns].values.sum(axis=1)
        data[self.new_col] = np.where(condition == 0, 0, 1)

        return data

class FracFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, columns, new_col):
        self.columns = columns
        self.new_col = new_col

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        data = X.copy()
        data[self.new_col] = data[self.columns].values % 1

        return data

class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns, strategy):
        self.columns  = columns        
        self.strategy = strategy

    def fit(self, X, y=None):        
        return self
    
    def transform(self, X):
        data = X.copy()            

        if self.strategy == 'mode':            
            data[self.columns].fillna(data[self.columns].mode()[0], inplace=True)

        elif self.strategy == 'mean':            
            data[self.columns].fillna(data[self.columns].mean(), inplace=True)

        elif self.strategy == 'median':
            data[self.columns].fillna(data[self.columns].median(), inplace=True)

        return data  

class Subtract(BaseEstimator, TransformerMixin):
    def __init__(self, col1, col2, new_col):
        self.col1 = col1
        self.col2 = col2
        self.new_col = new_col

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        data = X.copy()
        data[self.new_col] = data[self.col1] - data[self.col2]

        return data
