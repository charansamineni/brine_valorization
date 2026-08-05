import pandas as pd

# Option A: Read compressed CSV (Pandas auto-detects gzip compression)
#df = pd.read_csv("costing_vs_recovery_70gL_2026-08-03_1522.csv")

# diluate_vel = df[
#     df["Variable_Name"].str.contains("total_capital_cost", case=False, na=False)
# ]

# Select relevant columns for clear viewing
# columns_to_show = [
#     "NaCl_Recovery",
#     "Variable_Name",
#     "Value",
#     "Lower_Bound",
#     "Upper_Bound",
#     "Is_Fixed",
# ]
# diluate_vel_clean = diluate_vel[columns_to_show]

# Clean the first column name
#df.columns = df.columns.str.replace("#", "", regex=False).str.strip()

# Find the exact column containing "total_capital_cost"
# col = [c for c in df.columns if "total_capital_cost" in c][0]

# # Print recovery alongside total capital cost
# #print(df[["NaCl Recovery", col]])
# print(df[[col]])

df_all = pd.read_csv("updated_costing_vs_recovery_150gL_2026-08-04_1610.csv")

# print(df_all[[ "fs.costing.total_capital_cost", "fs.costing.total_operating_cost", "fs.costing.capital_recovery_factor", "fs.total_product_water", "fs.costing.utilization_factor" ]])
print(df_all[[ "Total Capital Cost", "Total Operating Cost", "Capital Recovery Factor", "Total Product Water", "Utilization Factor" ]])

dupes = df_all.columns[df_all.columns.duplicated()].tolist()
# if dupes:
#     print("Duplicate columns found:", dupes)

# target_vals = [1.470398e6, 2.176142e6]
# for col in df_all.columns:
#     series = df_all.loc[:, col]
#     if isinstance(series, pd.DataFrame):
#         series = series.iloc[:, 0]   # duplicate label — just take the first
#     if series.round(0).isin([round(v) for v in target_vals]).any():
#         print(col)