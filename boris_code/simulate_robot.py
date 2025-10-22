import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import os

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
# --- NEW: Penalty for when no APs are available at all ---
NO_AP_PENALTY = 2000

# AP positions - crucial for plotting them
AP_POSITIONS = {
    'ap1': (0, 0),       # Top-Left (x, y)
    'ap2': (SIDE1, 0),   # Top-Right
    'ap3': (0, SIDE2),   # Bottom-Left
    'ap4': (SIDE1, SIDE2) # Bottom-Right
}

# -------------------------------------------------------------------
# --- HELPER FUNCTIONS (No changes here)                          ---
# -------------------------------------------------------------------

def get_map_dims():
    zones_x = SIDE1 // ZONE_SIZE
    zones_y = SIDE2 // ZONE_SIZE
    return zones_x, zones_y

def zone_to_coords(zone_id, zones_x):
    if zone_id < 1:
        raise ValueError("Zone ID must be 1 or greater.")
    row = (zone_id - 1) // zones_x
    col = (zone_id - 1) % zones_x
    return (row, col)

def coords_to_zone(row, col, zones_x):
    return row * zones_x + col + 1

# -------------------------------------------------------------------
# --- TRAJECTORY GENERATION LOGIC (No changes here)               ---
# -------------------------------------------------------------------

# Cambia direzione quando sbatte contro il muro
def generate_trajectory_by_length(start_zone, length, zones_x, zones_y):
    path = [start_zone]
    current_row, current_col = zone_to_coords(start_zone, zones_x)
    DIRECTIONS = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}

    def get_valid_moves(r, c):
        valid = []
        for direction, (dr, dc) in DIRECTIONS.items():
            if 0 <= r + dr < zones_y and 0 <= c + dc < zones_x:
                valid.append((dr, dc))
        return valid

    current_direction = random.choice(get_valid_moves(current_row, current_col))

    for _ in range(length - 1):
        next_row = current_row + current_direction[0]
        next_col = current_col + current_direction[1]
        is_wall_hit = not (0 <= next_row < zones_y and 0 <= next_col < zones_x)

        # This is to avoid going back on the same path after a wall hit
        if is_wall_hit:
            possible_moves = get_valid_moves(current_row, current_col)
            reverse_direction = (-current_direction[0], -current_direction[1])
            non_reversing_moves = [m for m in possible_moves if m != reverse_direction]
            current_direction = random.choice(non_reversing_moves) if non_reversing_moves else reverse_direction

        current_row += current_direction[0]
        current_col += current_direction[1]
        path.append(coords_to_zone(current_row, current_col, zones_x))
        
    return path

# -------------------------------------------------------------------
# --- SIMULATION CORE (MODIFIED)                                  ---
# -------------------------------------------------------------------

def simulate_trajectory(df, path):
    total_latency = 0.0
    current_ap = None
    handover_events = []
    simulation_log = []
    rssi_cols = ['rssi_ap1', 'rssi_ap2', 'rssi_ap3', 'rssi_ap4']

    # --- Handle the starting zone ---
    start_zone = path[0]
    zone_sample = df[df['zone_id'] == start_zone].sample(n=1).iloc[0]
    
    # MODIFIED: Find the best AP that is actually available
    available_aps = {col.replace('rssi_', ''): zone_sample[col] for col in rssi_cols if zone_sample[col] >= RSSI_THRESHOLD}
    
    if available_aps:
        # Connect to the best available AP
        best_ap_name = max(available_aps, key=available_aps.get)
        current_ap = best_ap_name
        initial_latency = zone_sample[f'latency_{current_ap}']
        initial_rssi = zone_sample[f'rssi_{current_ap}']
        total_latency += initial_latency
        event = "Initial Connection"
    else:
        # MODIFIED: No APs available at the start
        current_ap = None
        initial_latency = NO_AP_PENALTY
        initial_rssi = -np.inf
        total_latency += initial_latency
        event = "No APs available at start"
        
    simulation_log.append({
        "zone": start_zone, "connected_ap": current_ap, 
        "latency_ms": initial_latency, "event": event,
        "rssi_dbm": initial_rssi
    })
    
    # --- Simulate the rest of the trajectory ---
    for zone_id in path[1:]:
        zone_sample = df[df['zone_id'] == zone_id].sample(n=1).iloc[0]
        
        # Check if currently connected AP is still good enough
        is_current_ap_ok = current_ap and zone_sample[f'rssi_{current_ap}'] >= RSSI_THRESHOLD

        if is_current_ap_ok:
            # Stay connected
            latency_this_step = zone_sample[f'latency_{current_ap}']
            total_latency += latency_this_step
            simulation_log.append({
                "zone": zone_id, "connected_ap": current_ap,
                "latency_ms": latency_this_step, "event": "Stay",
                "rssi_dbm": zone_sample[f'rssi_{current_ap}']
            })
        else:
            # Disconnected or need to find a new AP
            # MODIFIED: Find the best NEW available AP
            available_aps_new = {col.replace('rssi_', ''): zone_sample[col] for col in rssi_cols if zone_sample[col] >= RSSI_THRESHOLD}
            
            if available_aps_new:
                # Handover is possible
                new_best_ap = max(available_aps_new, key=available_aps_new.get)
                latency_of_new_ap = zone_sample[f'latency_{new_best_ap}']
                rssi_of_new_ap = zone_sample[f'rssi_{new_best_ap}']
                
                latency_this_step = HANDOVER_PENALTY + latency_of_new_ap
                total_latency += latency_this_step
                
                event_details = f"Handover from {current_ap} to {new_best_ap}"
                handover_events.append({"zone": zone_id, "from": current_ap, "to": new_best_ap})
                simulation_log.append({
                    "zone": zone_id, "connected_ap": new_best_ap,
                    "latency_ms": latency_this_step, "event": event_details,
                    "rssi_dbm": rssi_of_new_ap
                })
                current_ap = new_best_ap
            else:
                # MODIFIED: No APs available to connect to
                latency_this_step = NO_AP_PENALTY
                total_latency += latency_this_step
                event_details = f"Disconnected from {current_ap}, no new AP found"
                simulation_log.append({
                    "zone": zone_id, "connected_ap": None,
                    "latency_ms": latency_this_step, "event": event_details,
                    "rssi_dbm": -np.inf
                })
                current_ap = None

    avg_latency = total_latency / len(path)
    return avg_latency, simulation_log, handover_events

# -------------------------------------------------------------------
# --- VISUALIZATION (No changes here)                             ---
# -------------------------------------------------------------------

def plot_trajectory(path, handover_zones, zones_x, zones_y):
    start_zone, end_zone = path[0], path[-1]
    path_coords = [zone_to_coords(z, zones_x) for z in path]
    path_rows, path_cols = zip(*path_coords)

    fig, ax = plt.subplots(figsize=(zones_x * 1.2, zones_y * 1.2))
    
    # Draw zones and their ID numbers
    for i in range(zones_y):
        for j in range(zones_x):
            zone_id = coords_to_zone(i, j, zones_x)
            rect = patches.Rectangle((j, i), 1, 1, linewidth=1, edgecolor='k', facecolor='none')
            ax.add_patch(rect)
            ax.text(j + 0.5, i + 0.5, str(zone_id), ha='center', va='center', fontsize=10, color='gray')

    # Highlight Start (Green), End (Red), and Handover (Blue) zones
    start_row, start_col = zone_to_coords(start_zone, zones_x)
    ax.add_patch(patches.Rectangle((start_col, start_row), 1, 1, color='green', alpha=0.5, label='Start Zone'))
    
    end_row, end_col = zone_to_coords(end_zone, zones_x)
    ax.add_patch(patches.Rectangle((end_col, end_row), 1, 1, color='red', alpha=0.5, label='End Zone'))

    # Plot Handover zones
    for zone in handover_zones:
        h_row, h_col = zone_to_coords(zone, zones_x)
        ax.add_patch(patches.Rectangle((h_col, h_row), 1, 1, color='blue', alpha=0.5, label='_nolegend_')) # Label for legend will be added below
    
    # Plot the trajectory path
    ax.plot([c + 0.5 for c in path_cols], [r + 0.5 for r in path_rows], 
            marker='o', linestyle='-', color='black', label='Robot Path', markersize=6)

    # Plot AP locations
    ap_plot_coords = {
        'ap1': (AP_POSITIONS['ap1'][0] / ZONE_SIZE, AP_POSITIONS['ap1'][1] / ZONE_SIZE),
        'ap2': (AP_POSITIONS['ap2'][0] / ZONE_SIZE, AP_POSITIONS['ap2'][1] / ZONE_SIZE),
        'ap3': (AP_POSITIONS['ap3'][0] / ZONE_SIZE, AP_POSITIONS['ap3'][1] / ZONE_SIZE),
        'ap4': (AP_POSITIONS['ap4'][0] / ZONE_SIZE, AP_POSITIONS['ap4'][1] / ZONE_SIZE)
    }

    for ap_name, (ap_x_grid, ap_y_grid) in ap_plot_coords.items():
        adjusted_x = ap_x_grid
        adjusted_y = ap_y_grid

        if ap_x_grid == 0: adjusted_x = 0.5
        elif ap_x_grid == zones_x: adjusted_x = zones_x - 0.5
        
        if ap_y_grid == 0: adjusted_y = 0.5
        elif ap_y_grid == zones_y: adjusted_y = zones_y - 0.5

        ax.plot(adjusted_x, adjusted_y, 'P', markersize=15, color='purple', alpha=0.8, markeredgecolor='black', label=f'AP {ap_name[-1]}')
        ax.text(adjusted_x + 0.1, adjusted_y - 0.2, f'AP{ap_name[-1]}', color='darkslategray', fontsize=10, ha='left', va='bottom')
    
    ax.set_xlim(0, zones_x)
    ax.set_ylim(zones_y, 0) # Invert y-axis to have (0,0) at top-left
    ax.set_xticks(np.arange(zones_x + 1))
    ax.set_yticks(np.arange(zones_y + 1))
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Robot Trajectory and Wi-Fi Handovers (APs in Purple)", fontsize=14)
    ax.set_xlabel("X-coordinate (Zone Units)")
    ax.set_ylabel("Y-coordinate (Zone Units)")
    
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = {}
    for i, label in enumerate(labels):
        if label == '_nolegend_':
            if 'Handover Zones' not in unique_labels:
                unique_labels['Handover Zones'] = handles[i]
        else:
            unique_labels[label] = handles[i]
    
    handover_patch = patches.Rectangle((0, 0), 1, 1, color='blue', alpha=0.5)
    unique_labels['Handover Zones'] = handover_patch

    ax.legend(unique_labels.values(), unique_labels.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------
# --- MAIN EXECUTION BLOCK (No changes here)                      ---
# -------------------------------------------------------------------

if __name__ == '__main__':
    if not os.path.exists(INPUT_FILENAME):
        print(f"❌ Error: Dataset file '{INPUT_FILENAME}' not found.")
    else:
        print(f"Loading dataset from '{INPUT_FILENAME}'...")
        df = pd.read_csv(INPUT_FILENAME)
        zones_x, zones_y = get_map_dims()
        total_zones = zones_x * zones_y

        start_zone = random.randint(1, total_zones)
        robot_path = generate_trajectory_by_length(start_zone, TRAJECTORY_LENGTH, zones_x, zones_y)
        end_zone = robot_path[-1]
        
        avg_latency, log, handovers = simulate_trajectory(df, robot_path)
        
        print("\n" + "="*70)
        print("         🤖 ROBOT TRAJECTORY SIMULATION 🤖")
        print("="*70)
        print(f"🗺️  Map Dimensions: {zones_x}x{zones_y} zones ({total_zones} total)")
        print(f"🏁 Trajectory: Start Zone {start_zone} -> End Zone {end_zone}")
        print(f"👣 Path (Length {TRAJECTORY_LENGTH}): {robot_path}")
        print("\n--- Simulation Log ---")
        
        for entry in log:
            print(f"  - Zone {entry['zone']:<3}: "
                  f"Connected to {str(entry['connected_ap']):<4} | "
                  f"RSSI: {entry['rssi_dbm']:>6.2f} dBm | "
                  f"Latency: {entry['latency_ms']:>8.2f} ms | "
                  f"Event: {entry['event']}")

        print("\n--- Summary ---")
        print(f"🔄 Total Handovers: {len(handovers)}")
        print(f"⏱️  Average Latency Across Trajectory: {avg_latency:.2f} ms")
        print("="*70)

        handover_zone_ids = [h['zone'] for h in handovers]
        plot_trajectory(robot_path, handover_zone_ids, zones_x, zones_y)
