# Main imports ------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing


class my_preprocessing : 
    

    def handle_missing(df, method='fillna', fill_value=0, threshold=60, inplace=False, test_df=None, fill_numeric_only=False):
        """
        Calculate missing data percentage, impute missing values, and drop columns exceeding a missing value threshold.
        
        Parameters:
        - df: pandas DataFrame
        - method: str, method to handle missing values ('fillna', 'drop', 'interpolate'). Defaults to 'fillna'.
        - fill_value: value to fill with if method is 'fillna' (ex: ffill, bfill or a number). Defaults to 0.
        - threshold: float (in %), if a column has missing values above this percentage, it will be dropped. Defaults to 60(%).
        - inplace: bool, if True, modifies the DataFrame in place. Otherwise, returns a new DataFrame.
        - test_df: the test DataFrame to fill missing values using averages from the first DataFrame (df).

        Returns:
        - new_df: DataFrame with missing values handled based on specified method (if inplace=False).
        """

        # changed_rows = df[df.isnull().any(axis=1)]  # Rows with missing values in the original DataFrame

        if not inplace:
            df = df.copy()
    
        
        # Calculate percent of missing data
        missing_percent = df.isnull().mean() * 100
        missing_percent.sort_values(ascending=False, inplace=True)
        print("\nPercentage of missing data per column:\n", missing_percent)
        
        # Drop columns with missing percentage higher than the threshold
        columns_to_drop = missing_percent[missing_percent > threshold].index
        df.drop(columns=columns_to_drop, inplace=True)
        print(f"\nDropped columns with missing data > {threshold}%: {list(columns_to_drop)}")
        
        #  # Filter numeric columns if fill_numeric_only is True
        # if fill_numeric_only:
        #     numeric_columns = df.select_dtypes(include='number').columns
        # else:
        #     numeric_columns = df.columns

        # Handle missing data based on the specified method
        if method == 'fillna':
            df.fillna(fill_value, inplace=True)
        elif method == 'drop':
            df.dropna(inplace=True)
        elif method == 'interpolate':
            df.interpolate(inplace=True)
        elif method == 'ffill':
            df.ffill(inplace=True)  # Forward fill
            df.fillna(fill_value, inplace=True)
        elif method == 'bfill':
            df.bfill(inplace=True)  # Backward fill
            df.fillna(fill_value, inplace=True)
        # elif method == 'mean':
        #     df[numeric_columns].fillna(df[numeric_columns].mean(), inplace=True)  # Fill missing values with column means
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        
        ## print old and new rows
        # updated_rows = df.loc[changed_rows.index]  # Corresponding rows after changes
        # print("\nOriginal rowas with missing values:\n")
        # print(changed_rows.head(5))
        # print("\nRows with missing values that were updated:\n")
        # print(updated_rows.head(5))

        # if test_df is not None : 
        #     avg_values = df[numeric_columns].mean() # Calculate average values on the first DataFrame
        #     test_df[numeric_columns].fillna(avg_values, inplace=True)  # Fill missing values in the second DataFrame
        #     print("\nFilled missing values in the test df with averages from the train df.")
    
        
        if inplace:
            return None
        else:
            if test_df is not None:
                return df, test_df
            return df
        

    def scale_data(train_data, test_data=None):
        """
        Scale the train dataset using RobustScaler, 
        and optionally apply the same transformation to the test dataset.
        
        Parameters:
        - train_data (pd.DataFrame or np.ndarray): The training dataset to fit and transform.
        - test_data (pd.DataFrame or np.ndarray, optional): The test dataset to apply the same transformation.
        
        Returns:
        - train_data_scaled (pd.DataFrame or np.ndarray): Scaled training data.
        - test_data_scaled (pd.DataFrame or np.ndarray, optional): Scaled test data, if provided.
        """
        scaler = preprocessing.RobustScaler()
        cols = train_data.columns


        # Fit and transform the training data
        train_data_scaled = train_data.copy()
        train_data_scaled = pd.DataFrame(
        scaler.fit_transform(train_data), 
        columns=cols, 
        index=train_data.index) 

        # If test data is provided, transform it without refitting
        if test_data is not None:
            test_data_scaled = test_data.copy()
            test_data_scaled = pd.DataFrame(
            scaler.transform(test_data), 
            columns=cols, 
            index=test_data.index) 
            return train_data_scaled, test_data_scaled
        else:
            return train_data_scaled

    # Example usage
    # X_train_scaled, X_test_scaled = scale_data(X_train, X_test)




# Create a basic function to evaluate models
def evaluate_model(model):
    model.fit(train_choice, y_train_bin)
    y_pred_train = model.predict(train_choice)
    y_pred = model.predict(test_choice)
    cm = confusion_matrix(y_test_bin, y_pred, normalize='all')
    dispcm = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=model.classes_)

    # print(f"Accuracy on test: {accuracy_score(y_test_bin, y_pred)}")
    print(f"Accuracy on train: {accuracy_score(y_train_bin, y_pred_train)}")
    print(f"Classification Report:\n{classification_report(y_test_bin, y_pred)}")
    # print(f"AUC-ROC Score: {roc_auc_score(y_test_bin, model.predict_proba(test_choice)[:, 1])}")
    print(f"Confusion matrix:")
    dispcm.plot()

