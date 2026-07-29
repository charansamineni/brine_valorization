import pandas as pd

# Option A: Read compressed CSV (Pandas auto-detects gzip compression)
df = pd.read_csv("full_model_all_vars_150gL_20260729_154739.csv.gz")

diluate_vel = df[
    df["Variable_Name"].str.contains("velocity_basate", case=False, na=False)
]

# Select relevant columns for clear viewing
columns_to_show = [
    "NaCl_Recovery",
    "Variable_Name",
    "Value",
    "Lower_Bound",
    "Upper_Bound",
    "Is_Fixed",
]
diluate_vel_clean = diluate_vel[columns_to_show]

# Display unique diluate velocity variable names in your model
print("Diluate Velocity Variables Found:")
print(diluate_vel_clean["Variable_Name"].unique())

print("\nSample Data:")
print(diluate_vel_clean.head(10))