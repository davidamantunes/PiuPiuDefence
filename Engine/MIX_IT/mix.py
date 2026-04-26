import pandas as pd

temp1 = pd.read_csv("SimOutFINAL1.csv")
temp2 = pd.read_csv("SimOutFINAL2.csv")

# Append FINAL2 columns to the right of FINAL1.
combined = pd.concat([temp1.reset_index(drop=True), temp2.reset_index(drop=True)], axis=1)
combined.to_csv("SimOut.csv", index=False)
