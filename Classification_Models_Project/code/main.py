# --- Required Libraries ---
import pandas as pd
import time
from prettytable import PrettyTable
print("\n\t--- Classification Models ---")

# --- Import custom modules ---
from data_handling import Data
from decision_tree import Decision_Tree
from random_forest import Random_forest
from xg_boost import Xg_boost
table = PrettyTable()

# --- Model names for comparison table ---
models = ["Decision Tree","Random Forest","XGBoost"]

# --- Initialize data & model objects ---
data = Data()
dt = Decision_Tree()
time.sleep(1)
print("2.Unique_ID detected successful!")
rf = Random_forest()
time.sleep(1)
print("3.Target detected successful!")
xgb = Xg_boost()
time.sleep(1)
print("4.WOE & IV Calculated Successful!")
time.sleep(1)
print("5.VIF Calculated Successful! ")

# --- File Export Function ---
def file_saving(file_location):
    if file_location == {}:
        print("\nNo file saved , first run this model")
    else:
        # --- Display available files ---
        for x in file_location:
            print("--> ",x)

        # --- Ask user which file to export ---
        file_name = input("Enter correct file name to export : ")

        # --- Cross-check file and save file as Excel ---
        if file_name in file_location:
            temp = file_location[file_name]
            temp.to_excel(input("Enter file name to save as : ") + ".xlsx", index=False)
            print("\nFile exported successful!")
        else:
            print("\nFile not found!")


# --- Main Menu Loop ---
time.sleep(1)
condition = True
while condition :
    user_choice_1 = input("\nSelect option\n"
                          "1 - Run ML models\n"
                          "2 - View save data\n"
                          "3 - Exit\n"
                          "Enter your choice : ")
    # --- Run Models ---
    if user_choice_1 == "1":
        user_choice = input("\nWhich model do you want to run ?\n"
                            "1 - Decision Tree with single parameters\n"
                            "2 - Decision Tree with multiple parameters\n"
                            "3 - Random Forest with single parameters\n"
                            "4 - Random Forest with multiple parameters\n"
                            "5 - XGBoost with single parameters\n"
                            "6 - XGBoost with multiple parameters\n"
                            "7 - All Models with single parameter\n"
                            "Enter your choice(1-7) : ")

        # --- Decision Tree (Default) ---
        if user_choice == "1":
            if dt.dt_feature_list < 6:
                print("\nNot enough feature for prediction")
            else:
                dt.default_parameters()
                dt.dt_default_output()

        # --- Decision Tree (GridSearch) ---
        elif user_choice == "2":
            if dt.dt_feature_list < 6:
                print("\nNot enough feature for prediction")
            else:
                dt.grid_search()
                dt.dt_grid_output()

        # --- Random Forest (Default) ---
        elif user_choice == "3":
            if rf.rf_feature_list<5:
                print("\nNot enough feature for prediction")
            else:
                rf.default_parameters()
                rf.rf_default_output()

        # --- Random Forest (GridSearch) ---
        elif user_choice == "4":
            if rf.rf_feature_list<5:
                print("\nNot enough feature for prediction")
            else:
                rf.grid_search_rf()
                rf.rf_grid_output()

        # --- XGBoost (Default) ---
        elif user_choice == "5":
            xgb.xg_boost_data_reading()
            xgb.xgb_default_paramters()
            xgb.xg_default_output()

        # --- XGBoost (GridSearch) ---
        elif user_choice == "6":
            xgb.xg_boost_data_reading()
            xgb.grid_search_XGBoost()

        # --- Run All Models (Default Params) ---
        elif user_choice == "7":
            print()
            dt.default_parameters()
            rf.default_parameters()
            xgb.xg_boost_data_reading()
            xgb.xgb_default_paramters()

            # --- Collect metrics ---
            accuracy=[dt.default_accuracy,rf.rf_default_accuracy,xgb.xgb_default_accuracy]
            f1_score = [dt.default_f1,rf.rf_default_f1,xgb.xgb_default_f1]
            recall = [dt.default_recall,rf.rf_default_recall,xgb.xgb_default_recall]
            precision = [dt.default_precision,rf.rf_default_precision,xgb.xgb_default_precision]

            # --- Display comparison table ---
            table.add_column("Model_names",models)
            table.add_column("Accuracy",accuracy)
            table.add_column("F1_Score",f1_score)
            table.add_column("Recall",recall)
            table.add_column("Precision",precision)
            print(table)
            table.clear()

            # --- Save comparison results ---
            all_file = pd.DataFrame({
                "Model_name" : models,
                "Accuracy" : accuracy,
                "F1_Score" : f1_score,
                "Recall" : recall,
                "Precision" : precision
            })

            # ---  Store results in each model dictionary ---
            xgb.xgb_files["all_model_with single_parameter"]=all_file
            rf.rf_files["all_model_with single_parameter"]=all_file
            dt.dt_files["all_model_with single_parameter"]=all_file
        else :
            print("\nInvalid input!")

    # --- Export Saved Files ---
    elif user_choice_1 == "2":
        if dt.dt_files == {} and rf.rf_files == {} and xgb.xgb_files == {}:
            print("\nNo file saved,first run the model!")
        else:
            user_choice_2 = input("\nWhich file do you want to export ?\n"
                                  "1 - Decision Tree\n"
                                  "2 - Random Forest\n"
                                  "3 - XGBoost\n"
                                  "Enter your choice : ")
            if user_choice_2 == "1":
                file_saving(dt.dt_files)

            elif user_choice_2 == "2":
                file_saving(rf.rf_files)

            elif user_choice_2 == "3":
                file_saving(xgb.xgb_files)

    # --- Exit Program ---
    elif user_choice_1 == "3":
        print("\nExisting Goodbye...")
        condition = False
