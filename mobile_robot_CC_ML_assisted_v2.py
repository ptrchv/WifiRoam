import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Masking, Bidirectional, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------------------------------------------
# --- CONFIGURATION & HYPERPARAMETERS                             ---
# -------------------------------------------------------------------

INPUT_FILENAME = 'wifi_signal_dataset.csv'
SIDE1 = 120
SIDE2 = 60
ZONE_SIZE = 10

TRAJECTORY_LENGTH = 25
RSSI_THRESHOLD = -85.0
HANDOVER_PENALTY = 500
SWITCH_PENALTY = 50
NO_AP_PENALTY = 2000

# --- Model saving and loading configuration ---
# We will train a new, corrected model
LOAD_SAVED_MODEL = True
MODEL_FILENAME = 'robot_trajectory_model_corrected.keras'

# --- Seeding for reproducibility ---
SIMULATION_SEED = 2

# AP positions
AP_POSITIONS = {
    'ap1': (0, 0), 'ap2': (SIDE1, 0), 'ap3': (0, SIDE2), 'ap4': (SIDE1, SIDE2)
}

# -------------------------------------------------------------------
# --- HELPER FUNCTIONS (No changes)                               ---
# -------------------------------------------------------------------

def get_map_dims():
    return SIDE1 // ZONE_SIZE, SIDE2 // ZONE_SIZE

def zone_to_coords(zone_id, zones_x):
    if zone_id < 1: raise ValueError("Zone ID must be 1 or greater.")
    return (zone_id - 1) // zones_x, (zone_id - 1) % zones_x

def coords_to_zone(row, col, zones_x):
    return row * zones_x + col + 1

def generate_linear_trajectory(start_zone, length, zones_x, zones_y):
    path = [start_zone]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    current_direction = random.choice(directions)
    while len(path) < length:
        current_row, current_col = zone_to_coords(path[-1], zones_x)
        next_row, next_col = current_row + current_direction[0], current_col + current_direction[1]
        if 0 <= next_row < zones_y and 0 <= next_col < zones_x:
            path.append(coords_to_zone(next_row, next_col, zones_x))
        else:
            valid_directions = [d for d in directions if d != current_direction and 0 <= current_row + d[0] < zones_y and 0 <= current_col + d[1] < zones_x]
            if valid_directions: current_direction = random.choice(valid_directions)
            else: break
    return path

# -------------------------------------------------------------------
# --- DATA & MODEL FUNCTIONS (CORRECTED LOGIC)                    ---
# -------------------------------------------------------------------

def load_and_preprocess_data(filename, use_mean=False):
    print(f"Loading dataset from '{filename}'...")
    df = pd.read_csv(filename)
    for i in range(1, 5):
        df[f'latency_ap{i}'] = df[f'latency_ap{i}'].replace(-1, np.inf)
    if use_mean:
        print("Pre-processing data using mean values for model training.")
        return df.groupby('zone_id').mean().reset_index()
    else:
        print("Loading raw data samples for simulation.")
        return df

def prepare_live_trajectory_data(raw_df, robot_path):
    live_data = []
    for zone in robot_path:
        zone_samples = raw_df[raw_df['zone_id'] == zone]
        if not zone_samples.empty:
            live_data.append(zone_samples.sample(n=1, random_state=SIMULATION_SEED))
    return live_data

# CORRECTED: Model is simpler and only takes Zone IDs as input
def create_lstm_model(num_aps):
    """Creates a model that learns from sequences of Zone IDs."""
    model = Sequential([
        Masking(mask_value=0., input_shape=(None, 1)),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dense(num_aps, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# CORRECTED: Training data uses only Zone IDs as features
def prepare_data_for_lstm(df_mean, num_trajectories=2000, max_len=TRAJECTORY_LENGTH):
    """Generates training data where input X is the path of zone IDs."""
    print("Preparing training data (Input: Zone IDs, Output: Best AP)...")
    zones_x, zones_y = get_map_dims()
    total_zones = zones_x * zones_y
    X, y = [], []
    
    zone_data_lookup = {row.zone_id: row for index, row in df_mean.iterrows()}

    for _ in range(num_trajectories):
        start_zone = random.randint(1, total_zones)
        trajectory = generate_linear_trajectory(start_zone, max_len, zones_x, zones_y)
        optimal_aps = []
        for zone in trajectory:
            if zone in zone_data_lookup:
                zone_data = zone_data_lookup[zone]
                best_ap, min_latency = -1, np.inf
                for i in range(1, 5):
                    if getattr(zone_data, f'rssi_ap{i}') >= RSSI_THRESHOLD and getattr(zone_data, f'latency_ap{i}') < min_latency:
                        min_latency = getattr(zone_data, f'latency_ap{i}')
                        best_ap = i
                optimal_aps.append(best_ap - 1 if best_ap != -1 else 0)
        
        if len(optimal_aps) == len(trajectory):
            X.append(trajectory)
            y.append(optimal_aps)

    X_padded = pad_sequences(X, maxlen=max_len, padding='post', value=0., dtype='float32')
    y_padded = pad_sequences(y, maxlen=max_len, padding='post', value=-1)
    
    # Reshape X to be suitable for LSTM: (samples, timesteps, features)
    return X_padded.reshape(*X_padded.shape, 1), y_padded

# -------------------------------------------------------------------
# --- SIMULATION LOGIC (No changes needed here)                   ---
# -------------------------------------------------------------------

def simulate_trajectory_greedy(live_trajectory):
    total_latency, log, handovers, current_ap = 0, [], 0, None
    for i, zone_sample in enumerate(live_trajectory):
        if zone_sample.empty: continue
        zone_id = zone_sample.iloc[0]['zone_id']
        rssi_values = {f'ap{j}': zone_sample.iloc[0][f'rssi_ap{j}'] for j in range(1, 5)}
        latency_values = {f'ap{j}': zone_sample.iloc[0][f'latency_ap{j}'] for j in range(1, 5)}
        event, latency_this_step = "Move to new zone", 0
        best_ap_in_zone = max((ap for ap in rssi_values if rssi_values[ap] >= RSSI_THRESHOLD), key=rssi_values.get, default=None)
        if i == 0:
            if best_ap_in_zone: current_ap, latency_this_step, event = best_ap_in_zone, latency_values[best_ap_in_zone], "Initial connection"
            else: current_ap, latency_this_step, event = None, NO_AP_PENALTY, "No APs available at start"
        elif current_ap and rssi_values[current_ap] >= RSSI_THRESHOLD:
            latency_this_step = latency_values[current_ap]
        else:
            event = f"Disconnected from {current_ap}"; handovers += 1
            if best_ap_in_zone:
                current_ap, latency_this_step, event = best_ap_in_zone, HANDOVER_PENALTY + latency_values[best_ap_in_zone], event + f", reconnected to {current_ap}"
            else: current_ap, latency_this_step, event = None, NO_AP_PENALTY, event + ", no APs available"
        total_latency += latency_this_step
        log.append({'zone': zone_id, 'connected_ap': current_ap, 'rssi_dbm': rssi_values.get(current_ap, -np.inf), 'latency_ms': latency_this_step, 'event': event})
    return total_latency, log, handovers

def simulate_trajectory_nn(live_trajectory, robot_path, model):
    total_latency, log, handovers, current_ap = 0, [], 0, None
    
    # CORRECTED: Model predicts the entire strategy based only on the path of zone IDs
    path_padded = pad_sequences([robot_path], maxlen=TRAJECTORY_LENGTH, padding='post', value=0., dtype='float32')
    predictions = model.predict(path_padded.reshape(*path_padded.shape, 1), verbose=0)
    predicted_aps = np.argmax(predictions, axis=2)[0]

    for i, zone_sample in enumerate(live_trajectory):
        if zone_sample.empty: continue
        zone_id = zone_sample.iloc[0]['zone_id']
        rssi_values = {f'ap{j}': zone_sample.iloc[0][f'rssi_ap{j}'] for j in range(1, 5)}
        latency_values = {f'ap{j}': zone_sample.iloc[0][f'latency_ap{j}'] for j in range(1, 5)}
        event, latency_this_step = "Move to new zone", 0
        best_ap_in_zone = max((ap for ap in rssi_values if rssi_values[ap] >= RSSI_THRESHOLD), key=rssi_values.get, default=None)
        
        if i == 0:
            if best_ap_in_zone: current_ap, latency_this_step, event = best_ap_in_zone, latency_values[best_ap_in_zone], "Initial connection"
            else: current_ap, latency_this_step, event = None, NO_AP_PENALTY, "No APs available at start"
            recommended_ap = "N/A"
        else:
            # Use the pre-computed recommendation for this step
            recommended_ap = f'ap{predicted_aps[i] + 1}'
            is_current_ap_ok = current_ap and rssi_values[current_ap] >= RSSI_THRESHOLD
            
            if recommended_ap != current_ap and rssi_values.get(recommended_ap, -np.inf) >= RSSI_THRESHOLD:
                current_ap, latency_this_step, event, handovers = recommended_ap, SWITCH_PENALTY + latency_values[recommended_ap], f"Switched to {recommended_ap}", handovers + 1
            elif is_current_ap_ok:
                latency_this_step = latency_values[current_ap]
            else:
                event = f"Disconnected from {current_ap}"; handovers += 1
                if best_ap_in_zone:
                    current_ap, latency_this_step, event = best_ap_in_zone, HANDOVER_PENALTY + latency_values[best_ap_in_zone], event + f", reconnected to {current_ap}"
                else: current_ap, latency_this_step, event = None, NO_AP_PENALTY, event + ", no APs available"

        total_latency += latency_this_step
        log.append({'zone': zone_id, 'connected_ap': current_ap, 'rssi_dbm': rssi_values.get(current_ap, -np.inf), 
                    'latency_ms': latency_this_step, 'event': event, 'recommendation': recommended_ap})
    return total_latency, log, handovers

# -------------------------------------------------------------------
# --- VISUALIZATION (No changes)                                  ---
# -------------------------------------------------------------------

def plot_trajectory_and_map(robot_path, zones_x, zones_y, title):
    fig, ax = plt.subplots(figsize=(zones_x * 0.5, zones_y * 0.5))
    ax.set_title(title, fontsize=16)
    for r in range(zones_y):
        for c in range(zones_x):
            rect = patches.Rectangle((c * ZONE_SIZE, r * ZONE_SIZE), ZONE_SIZE, ZONE_SIZE, linewidth=1, edgecolor='black', facecolor='none', alpha=0.3)
            ax.add_patch(rect)
            ax.text(c * ZONE_SIZE + ZONE_SIZE / 2, r * ZONE_SIZE + ZONE_SIZE / 2, str(coords_to_zone(r,c,zones_x)), ha='center', va='center', fontsize=8)
    for ap, pos in AP_POSITIONS.items():
        ax.plot(pos[0], pos[1], 'r^', markersize=12, label=f'AP ({ap})' if ap not in plt.gca().get_legend_handles_labels()[1] else "")
        ax.text(pos[0], pos[1] + 2, ap, ha='center', va='bottom', fontsize=10)
    path_coords = [zone_to_coords(z, zones_x) for z in robot_path]
    path_x, path_y = [(c + 0.5) * ZONE_SIZE for r, c in path_coords], [(r + 0.5) * ZONE_SIZE for r, c in path_coords]
    ax.plot(path_x, path_y, 'b-o', markersize=5, linewidth=2, label='Robot Path')
    ax.plot(path_x[0], path_y[0], 'go', markersize=10, label='Start')
    ax.plot(path_x[-1], path_y[-1], 'ro', markersize=10, label='End')
    ax.set_xlim(-5, SIDE1 + 5); ax.set_ylim(-5, SIDE2 + 5)
    ax.set_aspect('equal', adjustable='box'); plt.gca().invert_yaxis()
    plt.legend(); plt.grid(True); plt.xlabel("X Coordinate"); plt.ylabel("Y Coordinate"); plt.show()

# -------------------------------------------------------------------
# --- MAIN EXECUTION                                              ---
# -------------------------------------------------------------------

if __name__ == "__main__":
    if not os.path.exists(INPUT_FILENAME):
        print(f"Error: Dataset file not found at '{INPUT_FILENAME}'.")
    else:
        random.seed(SIMULATION_SEED); np.random.seed(SIMULATION_SEED); tf.random.set_seed(SIMULATION_SEED)
        
        df_mean_for_training = load_and_preprocess_data(INPUT_FILENAME, use_mean=True)
        df_raw_for_simulation = load_and_preprocess_data(INPUT_FILENAME, use_mean=False)
        zones_x, zones_y = get_map_dims()
        total_zones, num_aps, model = zones_x * zones_y, 4, None

        if LOAD_SAVED_MODEL and os.path.exists(MODEL_FILENAME):
            print(f"\nLoading saved model from '{MODEL_FILENAME}'..."); model = load_model(MODEL_FILENAME)
            print("Model loaded successfully. Skipping training.")
        else:
            if LOAD_SAVED_MODEL: print(f"Model file '{MODEL_FILENAME}' not found. Training a new model.")
            
            X_data, y_data = prepare_data_for_lstm(df_mean_for_training)
            X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, random_state=SIMULATION_SEED)
            
            print("\nTraining the corrected LSTM model...")
            model = create_lstm_model(num_aps=num_aps)
            model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val), verbose=1)
            
            print(f"\nSaving trained model to '{MODEL_FILENAME}'..."); model.save(MODEL_FILENAME)
            print("Model saved successfully.")

        if model:
            start_zone = random.randint(1, total_zones)
            robot_path = generate_linear_trajectory(start_zone, TRAJECTORY_LENGTH, zones_x, zones_y)
            end_zone = robot_path[-1]
            live_trajectory_data = prepare_live_trajectory_data(df_raw_for_simulation, robot_path)
            
            greedy_latency, greedy_log, greedy_handovers = simulate_trajectory_greedy(live_trajectory_data)
            nn_latency, nn_log, nn_handovers = simulate_trajectory_nn(live_trajectory_data, robot_path, model)

            print("\n" + "="*80); print(" " * 25 + "🤖 ROBOT TRAJECTORY SIMULATION (GREEDY) 🤖"); print("="*80)
            print(f"Trajectory (Seed: {SIMULATION_SEED}): Start Zone {start_zone} -> End Zone {end_zone}\nPath (Length {len(robot_path)}): {robot_path}")
            print("\n--- Simulation Log (Greedy) ---")
            for entry in greedy_log:
                print(f"  - Zone {entry['zone']:<3}: Connected to {str(entry['connected_ap']):<4} | RSSI: {entry['rssi_dbm']:>6.2f} dBm | Latency: {entry['latency_ms']:>8.2f} ms | Event: {entry['event']}")
            print(f"\n--- Summary (Greedy) ---\n  - Total Latency: {greedy_latency:.2f} ms | Handovers: {greedy_handovers}")

            print("\n" + "="*95); print(" " * 32 + "🤖 ROBOT TRAJECTORY SIMULATION (NN) 🤖"); print("="*95)
            print(f"Trajectory (Seed: {SIMULATION_SEED}): Start Zone {start_zone} -> End Zone {end_zone}\nPath (Length {len(robot_path)}): {robot_path}")
            print("\n--- Simulation Log (NN) ---")
            for entry in nn_log:
                print(f"  - Zone {entry['zone']:<3}: Recommendation: {entry['recommendation']:<4} | Connected to {str(entry['connected_ap']):<4} | RSSI: {entry['rssi_dbm']:>6.2f} dBm | Latency: {entry['latency_ms']:>8.2f} ms | Event: {entry['event']}")
            print(f"\n--- Summary (NN) ---\n  - Total Latency: {nn_latency:.2f} ms | Handovers: {nn_handovers}")
            
            print("\n" + "="*80); print(" " * 32 + "🏆 FINAL COMPARISON 🏆"); print("="*80)
            print(f"  - Greedy Total Latency: {greedy_latency:.2f} ms\n  - NN Total Latency:     {nn_latency:.2f} ms")
            improvement = ((greedy_latency - nn_latency) / greedy_latency) * 100 if greedy_latency > 0 else 0
            print(f"\n  - The NN controller shows a {improvement:.2f}% improvement in latency for this trajectory.")
            
            plot_trajectory_and_map(robot_path, zones_x, zones_y, f"Robot Trajectory (Seed: {SIMULATION_SEED})")
