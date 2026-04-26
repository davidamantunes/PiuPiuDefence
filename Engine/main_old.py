from pathlib import Path

import pandas as pd


def load_simout(csv_path: Path) -> pd.DataFrame:
	"""Load SimOut.csv into a pandas DataFrame."""
	df = pd.read_csv(csv_path, skipinitialspace=True)

	df.columns = df.columns.str.strip()
	df["EnemyWeaponType"] = df["EnemyWeaponType"].astype(str).str.strip()
	df["ID"] = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")
	return df

def decision(simout_df: pd.DataFrame, corr_df: pd.DataFrame, threat_df: pd.DataFrame) -> pd.DataFrame:
	"""Create output with EnemyWeaponType, ID, and mapped DefenceType."""
	corr_df = corr_df.copy()
	corr_df.columns = corr_df.columns.str.strip()
	corr_df["EnemyWeaponType"] = corr_df["EnemyWeaponType"].astype(str).str.strip()
	corr_df["DefenceType"] = corr_df["DefenceType"].astype(str).str.strip()

	# Keep one row per weapon and ID from the corrected SimOut format.
	simout_unique = simout_df[["EnemyWeaponType", "ID"]].dropna(subset=["ID"]).drop_duplicates()

	output_df = pd.merge(simout_unique, corr_df, on="EnemyWeaponType", how="left")
	output_df = pd.merge(output_df, threat_df, on="ID", how="left")
	output_df = output_df.sort_values(by="ThreatLevel", ascending=False, na_position="last")
	return output_df[["EnemyWeaponType", "ID", "DefenceType", "ThreatLevel"]]

def preprocess_predictions(prediction_df: pd.DataFrame) -> pd.DataFrame:
	"""Clean and enrich predictions data for downstream use."""
	prediction_df = prediction_df.copy()
	prediction_df.columns = prediction_df.columns.str.strip()

	columns_to_drop = [
		"Px",
		"Py",
		"Speed",
		"Actual_Threat",
		"Actual_Damage",
	]
	prediction_df = prediction_df.drop(columns=columns_to_drop, errors="ignore")
	prediction_df = prediction_df.loc[:, ~prediction_df.columns.str.startswith("Unnamed:")]

	prediction_df["Predicted_Threat"] = pd.to_numeric(prediction_df["Predicted_Threat"], errors="coerce")
	prediction_df["Predicted_Damage"] = pd.to_numeric(prediction_df["Predicted_Damage"], errors="coerce")
	prediction_df["Predicted_Threat_x_Damage"] = prediction_df["Predicted_Threat"] * prediction_df["Predicted_Damage"]

	min_val = prediction_df["Predicted_Threat_x_Damage"].min(skipna=True)
	max_val = prediction_df["Predicted_Threat_x_Damage"].max(skipna=True)
	if pd.notna(min_val) and pd.notna(max_val) and max_val > min_val:
		prediction_df["Predicted_Threat_x_Damage"] = (
			(prediction_df["Predicted_Threat_x_Damage"] - min_val)
			/ (max_val - min_val)
		) * 10
	else:
		prediction_df["Predicted_Threat_x_Damage"] = 0.0

	return prediction_df

def create_predictions_output(prediction_df: pd.DataFrame) -> pd.DataFrame:
	"""Create output from preprocessed predictions, without raw predicted columns."""
	output_df = prediction_df.copy()
	output_df = output_df.drop(columns=["Predicted_Threat", "Predicted_Damage"], errors="ignore")
	return output_df.head(3)

def main() -> None:
	""""
	csv_path = Path(__file__).with_name("SimOut.csv")
	simout_df = load_simout(csv_path)

	csv_path = Path(__file__).with_name("Corr.csv")
	corr_df = pd.read_csv(csv_path)
	
	csv_path = Path(__file__).with_name("ThreatOut.csv")
	threat_df = pd.read_csv(csv_path)

	print("Loaded DataFrame:")
	print(simout_df)
	print(corr_df)
	print(threat_df)
	
	output = decision(simout_df, corr_df, threat_df)
	output_path = Path(__file__).with_name("out.csv")
	output.to_csv(output_path, index=False)

	print("Output DataFrame:")
	print(output)
	"""
	csv_path = Path(__file__).with_name("Predictions.csv")
	prediction_df = pd.read_csv(csv_path)
	prediction_df = preprocess_predictions(prediction_df)
	prediction_output_df = create_predictions_output(prediction_df)
	
	


if __name__ == "__main__":
	main()
