# --- Required Libraries ---
import time
from data_handling import Data
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection  import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import numpy as np
from prettytable import PrettyTable

# --- Initialize data object and perform preprocessing ---
data = Data()
data.data_reading()
data.woe_iv_calculations()
data.vif_calculations()

# -- Table for displaying model metrics --
xgb_table = PrettyTable()

# --- XGBoost Model Class ---
class Xg_boost:
    def __init__(self):
        """constructor for XGBoost"""
        self.point = data.count
        self.xgb_files = {}                                         # Stores all output files
        self.le = LabelEncoder()

        # --- Base dataframe & encoded target ---
        self.xg_boost_dataframe = data.dataframe
        self.xg_boost_target_df = self.le.fit_transform(data.target_df)

        # --- Number of selected features after IV & VIF ---
        self.xgb_features_list = len(data.selected_features)

    # --- Data Preparation for XGBoost ---
    def xg_boost_data_reading(self):

        # --- Remove target & unique ID ---
        self.xgb_features = [col for col in self.xg_boost_dataframe.columns if col != data.label and col != data.unique_id]
        self.xg_boost_dataframe=self.xg_boost_dataframe.drop(columns=[ col for col in self.xg_boost_dataframe.columns if col not in self.xgb_features])

        # --- Encode categorical columns ---
        for col in self.xg_boost_dataframe.columns:
            if self.xg_boost_dataframe[col].dtype == 'object':
                self.xg_boost_dataframe[col] = self.le.fit_transform(self.xg_boost_dataframe[col])

        # --- Train-test split with stratification ---
        self.xgb_sample_train, self.xgb_sample_test, self.xgb_target_train, self.xgb_target_test = train_test_split(
            self.xg_boost_dataframe, self.xg_boost_target_df , test_size=0.3, random_state=42, stratify=self.xg_boost_target_df
            )
        self.iteration = 0

    # --- GridSearchCV for XGBoost ---
    def grid_search_XGBoost(self):
        """XGBoost with GridSearchCV"""
        print("Running XGBoost with multiple parameters...")
        self.xg_boost_model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42
        )

        # --- Dynamic learning rate based on dataset size ---
        if len(self.xgb_sample_train)<1000:
            self.learing_rate = np.array([0.1,0.3])
        elif len(self.xgb_sample_train)>=1000 and len(self.xgb_sample_train)<10000:
            self.learing_rate = np.array([0.05,0.1])
        elif  len(self.xgb_sample_train)>10000 and len(self.xgb_sample_train)<100000 :
            self.learing_rate = np.array([0.01,0.05])
        else: 
            self.learing_rate = np.array([0.01,0.1])

        # --- Dynamic min_child_weight ---
        if len(self.xgb_sample_train)>=1000:
            self.min_child_weight = np.array([1,2])
        elif len(self.xgb_sample_train)>=1000 and len(self.xgb_sample_train)<10000:
            self.min_child_weight = np.array([1,3])
        elif  len(self.xgb_sample_train)>10000 and len(self.xgb_sample_train)<100000 :
            self.min_child_weight = np.array([5,7])
        else: 
            self.min_child_weight = np.array([12,20])

        # --- Hyperparameter grid ---
        self.xgb_param_grid = {
            "n_estimators": np.array([int(len(self.xgb_sample_train) * 0.01), int(len(self.xgb_sample_train) * 0.03),int(len(self.xgb_sample_train) * 0.05)]),
            "max_depth": np.array([3,5,7]),
            "learning_rate": self.learing_rate ,
            "min_child_weight" : self.min_child_weight,
            "subsample": [0.8],
        }

        # --- Multiple evaluation metrics ---
        scoreboard = {
            "accuracy": "accuracy",
            "f1": "f1",
            "recall": "recall",
            "precision": "precision"
        }

        # --- GridSearchCV configuration ---
        self.xgb_grid = GridSearchCV(
            estimator=self.xg_boost_model,
            param_grid=self.xgb_param_grid,
            scoring=scoreboard,
            refit="recall",       # final model selected based on recall   
            cv=4,             # cross validation
            n_jobs=-1,
        )

        # --- Train grid search ---
        self.xgb_grid.fit(self.xgb_sample_train,self.xgb_target_train)
        self.xgb_results = pd.DataFrame(self.xgb_grid.cv_results_)

        # print(self.xgb_results.isna().sum())
        # print(self.xgb_results.dtypes)

        # --- Best model & predictions ---
        self.best_model = self.xgb_grid.best_estimator_
        self.xgb_pred = self.best_model.predict(self.xgb_sample_test)

        # --- Evaluation metrics ---
        self.xgb_grid_accuracy = int(accuracy_score(self.xgb_target_test, self.xgb_pred) * 100)
        self.xgb_grid_f1 = int(f1_score(self.xgb_target_test, self.xgb_pred) * 100)
        self.xgb_grid_recall = int(recall_score(self.xgb_target_test, self.xgb_pred) * 100)
        self.xgb_grid_precision = int(precision_score(self.xgb_target_test, self.xgb_pred) * 100)

        # --- Data leakage detection ---
        if self.xgb_grid_accuracy == 100 and self.xgb_grid_recall == 100:
            print("\nModel have some data leakage , cleaning data with WOE,IV,VIF!")
            time.sleep(1)
            if self.xgb_features_list < 6:
                print("\nNot enough feature for prediction")
            else:
                self.xgb_sample_train = data.sample_train
                self.xgb_sample_test = data.sample_test
                self.xgb_target_train = data.target_train
                self.xgb_target_test = data.target_test
                self.xgb_features = data.selected_features
                self.iteration += 1

                if  self.iteration == 2  :
                     print("\nFinal result after data cleaning!")
                else :
                    self.grid_search_XGBoost()
                    self.output_grid()
        else:
            self.output_grid()

    # --- GridSearch Output & Saving ---
    def output_grid(self):
        self.xg_grid_output()

        # --- Select important GridSearch results ---
        self.xgb_final_results = self.xgb_results[
            ["param_n_estimators",
              "param_subsample",
              "param_min_child_weight",
              "param_learning_rate",
              "mean_test_accuracy",
              "mean_test_f1",
              "mean_test_recall",
              "mean_test_precision",
              "rank_test_recall"
              ]].copy()

        # --- Rename columns for clarity ---
        self.xgb_final_results.rename(columns={
            "mean_test_accuracy": "accuracy",
            "mean_test_f1": "f1_score",
            "param_min_samples_leaf": "min_samples_leaf",
            "param_min_samples_split": "min_samples_split",
            "param_max_depth": "max_depth",
            "mean_test_recall": "recall",
            "mean_test_precision": "precision",
            "rank_test_recall": "rank"
        }, inplace=True)

        # --- Feature importance ---
        self.xgb_feature_importance_gridsearchcv = pd.DataFrame({
            "Feature_names":  self.xgb_features,
            "Predicted_values": self.best_model.feature_importances_}).sort_values(by="Predicted_values",
                                                                              ascending=False)
   
        # self.xgb_final_results = self.xgb_final_results.style.apply(
        #     lambda row: ["background-color: lightgreen"] * len(row)
        #     if row["rank"] == 1
        #     else [""] * len(row),
        #     axis=1)

        # --- Save outputs ---
        self.xgb_files["XGB_feature_importance(GridSearchCV)"] = self.xgb_feature_importance_gridsearchcv
        self.xgb_files["XGB_GridSearchCV(CM)"] = self.xgb_final_results

    # --- Default Parameter XGBoost ---
    def xgb_default_paramters(self):
        print("Running XGBoost with single parameter...")
        time.sleep(2)
        data.woe_iv_calculations()
        data.vif_calculations()
        
        self.xgb_default_model = XGBClassifier(
            n_estimators=int(len(self.xgb_sample_train) * 0.07),
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            )

        # Train model
        self.xgb_default_model.fit(self.xgb_sample_train, self.xgb_target_train)
        self.xgb_default_pred = self.xgb_default_model.predict(self.xgb_sample_test)
        
        self.xgb_default_accuracy = int(accuracy_score(self.xgb_target_test, self.xgb_default_pred) * 100)
        self.xgb_default_f1 = int(f1_score(self.xgb_target_test, self.xgb_default_pred) * 100)
        self.xgb_default_recall = int(recall_score(self.xgb_target_test, self.xgb_default_pred) * 100)
        self.xgb_default_precision = int(precision_score(self.xgb_target_test, self.xgb_default_pred) * 100)
        
        if self.xgb_default_accuracy == 100 and self.xgb_default_recall == 100:
            print("\nModel have some data leakage , cleaning data with WOE,IV,VIF...")
            if self.xgb_features_list < 6:
                print("\nNot enough feature for prediction")
            else:
                self.xgb_sample_train = data.sample_train
                self.xgb_sample_test = data.sample_test
                self.xgb_target_train = data.target_train
                self.xgb_target_test = data.target_test
                self.xgb_features = data.selected_features

        else :

            self.xgboost_cm= pd.DataFrame({
                "n_estimators": int(len(self.xgb_sample_train) * 0.07),
                "learning_rate": 0.1,
                "max_depth": 5,
                "sub_sample":0.8,
                "Accuracy": self.xgb_default_accuracy,
                "F1_score":  self.xgb_default_f1,
                "Recall": self.xgb_default_recall,
                "Precision": self.xgb_default_precision
            }, index=[0])

            # --- Feature importance ---
            self.xgb_feature_importance_default = pd.DataFrame({
                "Feature_names": self.xgb_features,
                "Predicted_values": self.xgb_default_model.feature_importances_}).sort_values(by="Predicted_values",
                                                                                          ascending=False)

            # --- Save results ---
            self.xgb_files["xgb_default_parameters(CM)"] = self.xgboost_cm
            self.xgb_files["xgb_feature_importance"] = self.xgb_feature_importance_default

    # --- Display Outputs ---
    def xg_grid_output(self):
        xgb_table.add_column("Accuracy",[self.xgb_grid_accuracy])
        xgb_table.add_column("F1_Score",[self.xgb_grid_f1])
        xgb_table.add_column("Recall",[self.xgb_grid_recall])
        xgb_table.add_column("Precision",[self.xgb_grid_precision])
        print(xgb_table)
        xgb_table.clear()

    def xg_default_output(self):
        xgb_table.add_column("Accuracy",[self.xgb_default_accuracy])
        xgb_table.add_column("F1_Score",[self.xgb_default_f1])
        xgb_table.add_column("Recall",[self.xgb_default_recall])
        xgb_table.add_column("Precision",[self.xgb_default_precision])
        print(xgb_table)
        xgb_table.clear()

