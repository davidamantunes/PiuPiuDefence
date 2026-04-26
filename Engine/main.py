from pathlib import Path
import argparse

import pandas as pd

def preprocess_predictions(prediction_df: pd.DataFrame) -> pd.DataFrame:
	"""Clean and enrich predictions data for downstream use."""
	prediction_df = prediction_df.copy()
	prediction_df.columns = prediction_df.columns.str.strip()

	columns_to_keep = ["ID", "UAV_ID", "Time", "Predicted_Threat", "Predicted_Damage"]
	prediction_df = prediction_df.loc[:, ~prediction_df.columns.str.startswith("Unnamed:")]
	for column in columns_to_keep:
		if column not in prediction_df.columns:
			prediction_df[column] = pd.NA
	prediction_df = prediction_df[columns_to_keep]

	prediction_df["Predicted_Threat"] = pd.to_numeric(prediction_df["Predicted_Threat"], errors="coerce")
	prediction_df["Predicted_Damage"] = pd.to_numeric(prediction_df["Predicted_Damage"], errors="coerce")
	prediction_df["Time"] = pd.to_numeric(prediction_df["Time"], errors="coerce")

	return prediction_df

def create_predictions_output(prediction_df: pd.DataFrame, time_t: float) -> pd.DataFrame:
	"""Return all threats observed at the given time t."""
	output_df = prediction_df.copy()
	output_df = output_df[output_df["Time"] == time_t]
	raw_threat = output_df["Predicted_Threat"] * output_df["Predicted_Damage"]
	if raw_threat.empty:
		output_df["ThreatLevel"] = raw_threat
	else:
		min_threat = raw_threat.min(skipna=True)
		max_threat = raw_threat.max(skipna=True)
		if pd.isna(min_threat) or pd.isna(max_threat) or min_threat == max_threat:
			output_df["ThreatLevel"] = 0.0
		else:
			output_df["ThreatLevel"] = ((raw_threat - min_threat) / (max_threat - min_threat)) * 100.0
	output_df = output_df.sort_values(by="ThreatLevel", ascending=False, na_position="last")
	return output_df.reset_index(drop=True)

def main() -> None:
	parser = argparse.ArgumentParser(description="Get all threats at a specific time t.")
	parser.add_argument("--time", type=float, required=True, help="Time t to filter by.")
	args = parser.parse_args()

	csv_path = Path(__file__).with_name("SimOut.csv")
	prediction_df = pd.read_csv(csv_path)
	prediction_df = preprocess_predictions(prediction_df)
	prediction_output_df = create_predictions_output(prediction_df, args.time)
	
	print(prediction_output_df.to_csv(index=False))
	
if __name__ == "__main__":
	main()
