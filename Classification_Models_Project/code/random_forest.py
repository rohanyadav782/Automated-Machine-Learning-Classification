# --- importing libraries and data pipeline ---
import time
from data_handling import Data
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import numpy as np
from prettytable import PrettyTable

# --- Data Preparation ---
data = Data()
data.data_reading()
data.woe_iv_calculations()
data.vif_calculations()
rf_table = PrettyTable()

# --- Random Forest Model Class ---
class Random_forest:
    
    def __init__(self):
        self.point = data.count
        self.rf_files = {}                                  # Dictionary to store exportable results
        self.rf_feature_list = len(data.selected_features)  # Number of selected features
        
    def default_parameters(self):
        print("Running Random Forest with single parameter...")
        time.sleep(2)
        self.rf_model = RandomForestClassifier(
            n_estimators=int(len(data.sample_train) * 0.05),        # Number of trees
            max_depth=5,                                            # Controls tree depth
            min_samples_leaf=int(len(data.sample_train) * 0.02),
            min_samples_split=int(len(data.sample_train) * 0.05),
            random_state=42,
            n_jobs=1,class_weight='balanced'                        # Handles class imbalance
        )
        
        # --- Train the model ---
        self.rf_model.fit(data.sample_train, data.target_train)
        
        # --- Predict on test data ---
        self.rf_prediction = self.rf_model.predict(data.sample_test)
        
        # --- Calculate evaluation metrics ---
        self.rf_default_accuracy = int(accuracy_score(data.target_test, self.rf_prediction) * 100)
        self.rf_default_f1 = int(f1_score(data.target_test, self.rf_prediction,zero_division=0) * 100,)
        self.rf_default_recall = int(recall_score(data.target_test, self.rf_prediction,zero_division=0) * 100)
        self.rf_default_precision = int(precision_score(data.target_test, self.rf_prediction,zero_division=0) * 100)

        # --- Store model configuration and results ---
        self.random_forest_cm= pd.DataFrame({
            "min_samples_leaf": int(len(data.sample_train) * 0.02),
            "min_samples_split": int(len(data.sample_train) * 0.05),
            "max_depth": 5,
            "Accuracy": self.rf_default_accuracy,
            "F1_score":  self.rf_default_f1,
            "Recall": self.rf_default_recall,
            "Precision": self.rf_default_precision
        }, index=[0])


        # --- Extract feature importance --- 
        self.feature_importance_default_rf = pd.DataFrame({
            "Feature_names": data.selected_features,
            "Predicted_values": self.rf_model.feature_importances_}).sort_values(by="Predicted_values",ascending=False)
            
        # --- Save results for export ---                                                                                                                             
        self.rf_files["RF_default_parameters(CM)"] = self.random_forest_cm
        self.rf_files["RF_feature_importance"] = self.feature_importance_default_rf
        
    def grid_search_rf(self):
        print("Running Random Forest with multiple parameters...")
        # --- Base model for grid search --- 
        self.rf_model_grid = RandomForestClassifier(random_state=42,n_jobs=-1,class_weight='balanced', )
        
        # --- Hyperparameter search space --- 
        self.rf_grid = {
            "n_estimators" : np.array([int(len(data.sample_train) * 0.04),int(len(data.sample_train) * 0.01),int(len(data.sample_train) * 0.02)]),
            "max_depth" : np.array([7,5]),
            "min_samples_split":np.array([int(len(data.sample_train) * 0.07),int(len(data.sample_train) * 0.09),int(len(data.sample_train) * 0.05)]),
            "min_samples_leaf" : np.array([int(len(data.sample_train) * 0.07),int(len(data.sample_train) * 0.05),int(len(data.sample_train) * 0.02)])}
        
        # --- Multiple scoring metrics --- 
        scoreboard = {
            "accuracy": "accuracy",
            "f1": "f1",
            "recall": "recall",
            "precision": "precision"
        }
        
        # --- GridSearch with recall as primary metric --- 
        self.rf_grid_search_cv = GridSearchCV(
            estimator = self.rf_model_grid,
            scoring = scoreboard,
            param_grid=self.rf_grid,
            cv=4,
            refit="recall")
            
        # --- Train GridSearch model --- 
        self.rf_grid_search_cv.fit(data.sample_train, data.target_train)
        
        # --- Store cross-validation results --- 
        self.results_df = pd.DataFrame(self.rf_grid_search_cv.cv_results_)
        
        # --- Best model from GridSearch --- 
        best_model = self.rf_grid_search_cv.best_estimator_
        
        # --- Predict using best model --- 
        self.rf_pred = best_model.predict(data.sample_test)
        
        # --- Evaluation metrics --- 
        self.rf_grid_accuracy = int(accuracy_score(data.target_test, self.rf_pred) * 100)
        self.rf_grid_f1 = int(f1_score(data.target_test, self.rf_pred) * 100)
        self.rf_grid_recall = int(recall_score(data.target_test, self.rf_pred) * 100)
        self.rf_grid_precision = int(precision_score(data.target_test, self.rf_pred) * 100)

        # --- Final summarized results --- 
        self.rf_final_results = self.results_df[
            ["param_min_samples_split",
             "param_min_samples_leaf",
             "param_max_depth",
             "mean_test_accuracy",
             "mean_test_f1",
             "mean_test_recall",
             "mean_test_precision",
             "rank_test_accuracy"
             ]].copy()
        
        # --- Rename columns for clarity --- 
        self.rf_final_results.rename(columns={
            "mean_test_accuracy": "accuracy",
            "mean_test_f1": "f1_score",
            "param_min_samples_leaf": "min_samples_leaf",
            "param_min_samples_split": "min_samples_split",
            "param_max_depth": "max_depth",
            "mean_test_recall": "recall",
            "mean_test_precision": "precision",
            "rank_test_accuracy": "rank"
        }, inplace=True)

        # --- Feature importance from optimized model ---    
        self.rf_feature_importance_gridsearchcv = pd.DataFrame({
            "Feature_names": data.selected_features,
            "Predicted_values": best_model.feature_importances_}).sort_values(by="Predicted_values",
                                                                              ascending=False)
   
        # --- Highlight best-ranked configuration --- 
        self.rf_updated_df = self.rf_final_results.style.apply(
            lambda row: ["background-color: lightgreen"] * len(row)
            if row["rank"] == 1
            else [""] * len(row),
            axis=1)

        # --- Save GridSearch results --- 
        self.rf_files["RF_GridSearchCV(CM)"] = self.rf_updated_df
        self.rf_files["RF_feature_importance(GridSearchCV)"] = self.rf_feature_importance_gridsearchcv

    def rf_default_output(self):
        rf_table.add_column("Accuracy",[self.rf_default_accuracy])
        rf_table.add_column("F1_Score",[self.rf_default_f1])
        rf_table.add_column("Recall",[self.rf_default_recall])
        rf_table.add_column("Precision",[self.rf_default_precision])
        print(rf_table)
        rf_table.clear()

    def rf_grid_output(self):
        rf_table.add_column("Accuracy", [self.rf_grid_accuracy])
        rf_table.add_column("F1_Score", [self.rf_grid_f1])
        rf_table.add_column("Recall", [self.rf_grid_recall])
        rf_table.add_column("Precision", [self.rf_grid_precision])
        print(rf_table)
        rf_table.clear()
