# --- importing library ---
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

# --- file path ---
file = input("\nEnter your file path : ")
unique = input("Enter unique_ID : ")        # unique ID
label = input("Enter target : ")            # target/label
sp = file.split()                           # spilt to indentify file extension

print("\n1.Reading file...")
score = 0
if sp[-1][-4:] == "xlsx":                   # read xlsx file
    df2 = pd.read_excel(rf"{file}")
    score += 1
elif sp[-1][-3:] == "csv":                  # read csv file
    df2 = pd.read_csv(rf"{file}")
    score += 1
else:
    print("Invalid path")

dependent_feature = input("Any dependent feature in dataset ?\n"
                                    "Enter your choice(yes/no) : ").lower()
if dependent_feature == "yes":
        no_feature = int(input("How many dependent feature in dataset : "))
        independent_feature_list = []
        for x in range(no_feature):
            name = input(f"{x}. Enter correct feature name : ")
            independent_feature_list.append(name)

# --- Data Preprocessing ---
class Data:
    # --- Data Reading & Validation ---
    def data_reading(self):
        if dependent_feature == "yes":
            self.dataframe = df2.drop(columns=independent_feature_list)
        else:
            self.dataframe = df2
        self.unique_id = unique

        # --- Check if unique ID exists ---
        if self.unique_id in self.dataframe.columns:
            self.label = label
            self.count = score
            self.count += 1

            # --- Check if target column exists ---
            if self.label in self.dataframe.columns:
                self.count += 1
                self.target_df = self.dataframe[self.label]
                return self.dataframe

        else:
            return None

    # WOE Calculation
    def woe_iv_calculations(self):
        """ calculating WOE & IV for main dataframe"""

        def calculation(data, feature, target):
            df = data[[feature, target]].copy()

            # Total good & bad
            total_good = (df[target] == 0).sum()
            total_bad = (df[target] == 1).sum()

            # Group by feature
            grouped = df.groupby(feature)

            woe_df = grouped[target].agg(
                total='count',
                bad='sum'
            ).reset_index()

            # Good = Total - Bad
            woe_df['good'] = woe_df['total'] - woe_df['bad']

            # Distribution
            woe_df['dist_good'] = woe_df['good'] / total_good
            woe_df['dist_bad'] = woe_df['bad'] / total_bad

            # Handle zero division using smoothing
            woe_df['dist_good'] = woe_df['dist_good'].replace(0, 0.0001)
            woe_df['dist_bad'] = woe_df['dist_bad'].replace(0, 0.0001)

            # WOE calculation
            woe_df['WOE'] = np.log(woe_df['dist_good'] / woe_df['dist_bad'])

            # IV calculation
            woe_df['IV'] = (woe_df['dist_good'] - woe_df['dist_bad']) * woe_df['WOE']

            return woe_df, woe_df['IV'].sum()

        try:
            # --- Select features excluding target & unique ID ---
            self.features = [col for col in self.dataframe.columns if col != self.label and col != self.unique_id]
            # print(self.features)
        except ValueError:
            print("Insertion Error")
        self.iv_summary = []

        # --- Calculate IV for each feature ---
        for feature in self.features:
            woe_table, iv_value = calculation(self.dataframe, feature, self.label)
            self.iv_summary.append({
                'Feature': feature,
                'IV': iv_value
            })

        # --- Create IV summary dataframe ---
        self.iv_df = pd.DataFrame(self.iv_summary).sort_values(by="IV", ascending=False)

        # --- Feature selection based on IV thresholds ---
        self.selected_features = self.iv_df.loc[
            (self.iv_df["IV"] > 0.02) & (self.iv_df["IV"] < 0.50),
            'Feature'
        ].tolist()

    # --- VIF Calculations ---
    def vif_calculations(self):
        """ calculations vif with dataframe """

        # --- Store unique ID separately ---
        self.unique_df = self.dataframe[self.unique_id]

        # --- Keep only IV-selected features ---
        self.dataframe_copy_2 = self.dataframe.drop(
            columns=[col for col in self.dataframe.columns if col not in self.selected_features])

        # --- Encode target variable ---
        le = LabelEncoder()
        self.target_df = le.fit_transform(self.target_df)

        # --- Encode categorical features ---
        for col in self.dataframe_copy_2.columns:
            if self.dataframe_copy_2[col].dtype == 'object':
                self.dataframe_copy_2[col] = le.fit_transform(self.dataframe_copy_2[col])

        # --- Prepare dataframe for VIF ---
        main_DF_vif = self.dataframe_copy_2.copy()
        main_DF_vif = main_DF_vif.replace([np.inf, -np.inf], np.nan)
        main_DF_vif = main_DF_vif.fillna(0)

        # --- Add constant for VIF calculation ---
        main_DF_vif['const'] = 1

        # --- Calculate VIF values ---
        vif_df = pd.DataFrame()
        vif_df['Feature'] = main_DF_vif.columns
        vif_df['VIF'] = [
            variance_inflation_factor(main_DF_vif.values, i)
            for i in range(main_DF_vif.shape[1])
        ]
        # --- Drop constant column ---
        vif_df = vif_df[vif_df['Feature'] != 'const']

        # --- Select features with acceptable multicollinearity ---
        self.selected_features = vif_df.loc[
            (vif_df["VIF"] < 5.5),
            'Feature'
        ].tolist()

        # --- Final feature set ---
        self.dataframe_copy_2 = self.dataframe_copy_2.drop(
            columns=[col for col in self.dataframe_copy_2.columns if col not in self.selected_features])

        # --- Train-test split with stratification ---
        self.sample_train, self.sample_test, self.target_train, self.target_test, self.cust_train, self.cust_test = train_test_split(
            self.dataframe_copy_2, self.target_df, self.unique_df, test_size=0.3, random_state=42,
            stratify=self.target_df)
