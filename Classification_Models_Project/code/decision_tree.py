# --- Required Libraries ---
import time
from data_handling import Data
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import numpy as np
from prettytable import  PrettyTable

# --- Initialize data object and perform preprocessing ---
data = Data()
data.data_reading()
data.woe_iv_calculations()
data.vif_calculations()

# --- Table for displaying metrics ---
dt_table = PrettyTable()

# --- Decision Tree Model Class ---
class Decision_Tree:
    def __init__(self):
        """constructor for decision tree"""
        self.point = data.count
        self.dt_files = {}                              # Stores model outputs
        self.dt_feature_list = len(data.selected_features)

    # --- Default Parameters Model ---
    def default_parameters(self):
        """decision tree with default paramters"""
        print("Running Decision Tree with single parameter...")
        time.sleep(2)
        # --- Initialize model with balanced class weight --
        self.decision_tree = DecisionTreeClassifier(
            min_samples_leaf = int(len(data.sample_train) * 0.02),
            min_samples_split=int(len(data.sample_train) * 0.05),
            max_depth=5,
            criterion="gini",
            class_weight='balanced' )

        # --- Train model ---
        self.decision_tree.fit(data.sample_train, data.target_train)

        # --- Predictions ---
        self.prediction = self.decision_tree.predict(data.sample_test)

        # --- Evaluation metrics ---
        self.default_accuracy = int(accuracy_score(data.target_test, self.prediction) * 100)
        self.default_f1 = int(f1_score(data.target_test, self.prediction) * 100)
        self.default_recall = int(recall_score(data.target_test, self.prediction) * 100)
        self.default_precision = int(precision_score(data.target_test, self.prediction) * 100)

        # --- Store performance metrics ---
        self.decision_tree_cm = pd.DataFrame({
            "min_samples_leaf": int(len(data.sample_train) * 0.02),
            "min_samples_split": int(len(data.sample_train) * 0.05),
            "max_depth": 5,
            "criterion": "gini",
            "Accuracy": int(accuracy_score(data.target_test, self.prediction) * 100),
            "F1_score": int(f1_score(data.target_test, self.prediction) * 100),
            "Recall": int(recall_score(data.target_test, self.prediction) * 100),
            "Precision": int(precision_score(data.target_test, self.prediction) * 100)
        }, index=[0])

        # --- Feature importance ---
        self.feature_importance_default = pd.DataFrame({
            "Feature_names": data.selected_features,
            "Predicted_values": self.decision_tree.feature_importances_}).sort_values(by="Predicted_values",
                                                                                      ascending=False)

        # --- Save outputs ---
        self.dt_files["DT_default_parameters(CM)"] = self.decision_tree_cm
        self.dt_files["DT_feature_importance"] = self.feature_importance_default

    # --- GridSearchCV Multiple Parameter ---
    def grid_search(self):
        """decision tree with GridSearchCV"""
        print("Running Decision Tree with multiple parameters...")
        self.model = DecisionTreeClassifier(random_state=42,class_weight='balanced')

        # --- Hyperparameter grid ---
        self.grid_parameters = {
            "max_depth": np.array([4, 5, 6]),
            "min_samples_split": np.array([int(len(data.sample_train) * 0.14), int(len(data.sample_train) * 0.04),
                                           int(len(data.sample_train) * 0.1)]),
            "min_samples_leaf": np.array([int(len(data.sample_train) * 0.07), int(len(data.sample_train) * 0.02),
                                          int(len(data.sample_train) * 0.05)]),
            "criterion": np.array(["gini"])}

        # --- Multiple scoring metrics ---
        scoreboard = {
            "accuracy": "accuracy",
            "f1": "f1",
            "recall": "recall",
            "precision": "precision"
        }
        # --- GridSearchCV configuration ---
        self.grid = GridSearchCV(
            estimator=self.model,
            scoring=scoreboard,
            param_grid=self.grid_parameters,
            cv=4,
            refit="recall",
            n_jobs=-1)

        # --- Train grid search ---
        self.grid.fit(data.sample_train, data.target_train)
        self.results_df = pd.DataFrame(self.grid.cv_results_)

        # --- Best estimator ---
        best_model = self.grid.best_estimator_
        self.y_pred = best_model.predict(data.sample_test)
        # print("\n",classification_report(data.target_test, self.y_pred))

        # --- Evaluation metrics ---
        self.grid_accuracy = int(accuracy_score(data.target_test, self.y_pred) * 100)
        self.grid_f1 = int(f1_score(data.target_test, self.y_pred) * 100)
        self.grid_recall = int(recall_score(data.target_test, self.y_pred) * 100)
        self.grid_precision = int(precision_score(data.target_test, self.y_pred) * 100)

        # --- Extract important GridSearch results ---
        # self.final_results = pd.DataFrame({"model_iteration":[itr+1 for itr in range(len(self.results_df))]})
        self.final_results = self.results_df[
            ["param_min_samples_split",
             "param_min_samples_leaf",
             "param_max_depth",
             "mean_test_accuracy",
             "mean_test_f1",
             "mean_test_recall",
             "mean_test_precision",
             "rank_test_accuracy"
             ]].copy()

        # --- Rename columns for readability ---
        self.final_results.rename(columns={
            "mean_test_accuracy": "accuracy",
            "mean_test_f1": "f1_score",
            "param_min_samples_leaf": "min_samples_leaf",
            "param_min_samples_split": "min_samples_split",
            "param_max_depth": "max_depth",
            "mean_test_recall": "recall",
            "mean_test_precision": "precision",
            "rank_test_accuracy": "rank"
        }, inplace=True)

        # --- Feature importance from best model ---
        self.feature_importance_gridsearchcv = pd.DataFrame({
            "Feature_names": data.selected_features,
            "Predicted_values": best_model.feature_importances_}).sort_values(by="Predicted_values",
                                                                              ascending=False)
        # print(self.results_df)

        # --- Highlight best-ranked row ---
        self.updated_df = self.final_results.style.apply(
            lambda row: ["background-color: lightgreen"] * len(row)
            if row["rank"] == 1
            else [""] * len(row),
            axis=1)

        # --- Save outputs ---
        self.dt_files["DT_GridSearchCV(CM)"] = self.updated_df
        self.dt_files["DT_feature_importance(GridSearchCV)"] = self.feature_importance_gridsearchcv

    # --- Output Helpers ---
    def dt_default_output(self):
        dt_table.add_column("Accuracy", [self.default_accuracy])
        dt_table.add_column("F1_Score", [self.default_f1])
        dt_table.add_column("Recall", [self.default_recall])
        dt_table.add_column("Precision", [self.default_precision])
        print(dt_table)
        dt_table.clear()

    def dt_grid_output(self):
        dt_table.add_column("Accuracy", [self.grid_accuracy])
        dt_table.add_column("F1_Score", [self.grid_f1])
        dt_table.add_column("Recall", [self.grid_recall])
        dt_table.add_column("Precision", [self.grid_precision])
        print(dt_table)
        dt_table.clear()
