import csv
import os

def run_rule_based_decision():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    sim_data_path = os.path.join(base_dir, 'NN_Model', 'Dataset', 'SimOut_Test.csv')
    predictions_path = os.path.join(base_dir, 'NN_Model', 'Outputs', 'test_results_predictions.csv')
    out_path = os.path.join(base_dir, 'RuleBased_Decision', 'Defense_Decisions.csv')

    sim_data = []
    with open(sim_data_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sim_data.append(row)

    preds_data = []
    with open(predictions_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            preds_data.append(row)

    # Defenses definition with costs
    defenses_db = {
        'Geran2': [{'name': 'MobileFireGroup', 'cost': 1000},
                   {'name': 'InterceptorDrone', 'cost': 1500},
                   {'name': 'Bukovel-AD', 'cost': 500000},
                   {'name': 'Pokrova', 'cost': 1000000}],
        'Kalibr': [{'name': 'S-300', 'cost': 1500000},
                   {'name': 'Patriot', 'cost': 6000000}],
        'Kinzhal': [{'name': 'Patriot', 'cost': 6000000}]
    }

    results = []

    for i in range(len(sim_data)):
        sim_row = sim_data[i]
        pred_row = preds_data[i] if i < len(preds_data) else None

        weapon_type = sim_row.get('EnemyWeaponType', '')
        time_step = float(sim_row.get('Time', 0))
        uav_id = sim_row.get('UAV_ID', '')
        id_val = sim_row.get('ID', '')
        pos_x = sim_row.get('MeasuredX', '')
        pos_y = sim_row.get('MeasuredY', '')

        threat = float(pred_row.get('Predicted_Threat', 0)) if pred_row else 0
        damage = float(pred_row.get('Predicted_Damage', 0)) if pred_row else 0
        
        # Priority calculation
        priority = threat * damage
        
        available_defenses = defenses_db.get(weapon_type, [])
        best_defense = "None"
        
        if available_defenses:
            # Sort by cost ascending
            available_defenses = sorted(available_defenses, key=lambda x: x['cost'])
            best_defense = available_defenses[0]['name']  # Start with the cheapest defense
            
            # Simple rule: if predicted threat > 3 (very high), pick a more capable/expensive defense
            # (picking the most expensive one valid for that target)
            if threat > 3 and len(available_defenses) > 1:
                best_defense = available_defenses[-1]['name']

        results.append({
            'TimeStep': time_step,
            'ID': id_val,
            'UAV_ID': uav_id,
            'PositionX': pos_x,
            'PositionY': pos_y,
            'Defense Choice': best_defense,
            'Priority': priority
        })

    # Sort primarily by time step (ascending), then priority (descending within same timestep)
    #results = sorted(results, key=lambda x: (x['TimeStep'], -x['Priority']))

    with open(out_path, 'w', newline='') as f:
        fieldnames = ['TimeStep', 'ID', 'UAV_ID', 'PositionX', 'PositionY', 'Defense Choice', 'Priority']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Decisions successfully saved to {out_path}")

if __name__ == '__main__':
    run_rule_based_decision()
