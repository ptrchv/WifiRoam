import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import math
import random
from collections import deque, namedtuple
import time

# --- Hyperparameters ---
MAP_SIZE = 10
MAP_METERS = 40.0
TRAJECTORY_LEN = 15
LOOKAHEAD_STEPS = 3      # How many future steps to include in the state

BUFFER_SIZE = 20000
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 75000
TARGET_UPDATE = 10
N_EPISODES = 1500        # Increased episodes for the larger state space
LEARNING_RATE = 1e-4
N_EVAL_EPISODES = 1

# --- 1. The Environment Class ---
class WifiRLEnvironment:
    """
    Simulates the map, APs, and network conditions.
    The agent's state now includes all AP loads/RSSI and future trajectory steps.
    """
    def __init__(self, map_size=MAP_SIZE, map_meters=MAP_METERS, lookahead_steps=LOOKAHEAD_STEPS):
        self.map_size = map_size
        self.map_meters = map_meters
        self.lookahead_steps = lookahead_steps
        self.n_zones = map_size * map_size
        self.cell_size = self.map_meters / self.map_size
        self.n_aps = 4

        self.T_SWITCH = 10
        self.RSSI_DISCONNECT_THRESHOLD = -85

        self.ap_locations = {i+1: np.array(pos) for i, pos in enumerate([
            [0, 0], [0, self.map_meters], [self.map_meters, 0], [self.map_meters, self.map_meters]
        ])}
        self.base_ap_loads = {1: 0.4, 2: 0.4, 3: 0.4, 4: 0.4}

        self.robot_zone_id = 0
        self.current_ap = 1
        self.trajectory = []
        self.traj_step = 0

    def _get_coords_from_zone_id(self, zone_id):
        row, col = zone_id // self.map_size, zone_id % self.map_size
        x, y = (col + 0.5) * self.cell_size, (row + 0.5) * self.cell_size
        return np.array([x, y])

    def _calculate_rssi(self, zone_id, ap_id):
        robot_coords = self._get_coords_from_zone_id(zone_id)
        dist = np.linalg.norm(robot_coords - self.ap_locations[ap_id])
        path_loss_rssi = -30 - 35 * math.log10(dist + 1)
        noise = np.random.normal(0, 2)
        return path_loss_rssi + noise

    def _get_current_ap_loads(self):
        return {ap_id: max(0.1, min(1.0, base + np.random.uniform(-0.1, 0.1)))
                for ap_id, base in self.base_ap_loads.items()}

    def _calculate_latency(self, rssi, load):
        rssi_quality = 100 + rssi
        latency = 10 + (load * 100) + (150 / (rssi_quality + 5))
        return max(5, latency)

    def get_state(self):
        """
        Constructs the state vector for the agent.
        State = [current_zone, current_ap, all_loads, all_rssi, future_zone_1, future_zone_2, ...]
        """
        # Part 1: Current location and connection
        zone_vec = np.zeros(self.n_zones)
        zone_vec[self.robot_zone_id] = 1.0
        ap_vec = np.zeros(self.n_aps)
        ap_vec[self.current_ap - 1] = 1.0

        # Part 2: Full network scan at current location
        loads = self._get_current_ap_loads()
        load_vec = np.array([loads[i] for i in range(1, self.n_aps + 1)])
        rssi_vec = np.array([self._calculate_rssi(self.robot_zone_id, i) for i in range(1, self.n_aps + 1)])
        normalized_rssi_vec = (rssi_vec + 100) / 100.0 # Normalize

        # Part 3: Trajectory look-ahead
        future_trajectory_vecs = []
        for i in range(1, self.lookahead_steps + 1):
            future_step = self.traj_step + i
            future_zone_vec = np.zeros(self.n_zones)
            if future_step < len(self.trajectory):
                future_zone_id = self.trajectory[future_step]
                future_zone_vec[future_zone_id] = 1.0
            # If trajectory ends, the vector remains all zeros as padding
            future_trajectory_vecs.append(future_zone_vec)

        # Concatenate all parts into the final state vector
        state_parts = [zone_vec, ap_vec, load_vec, normalized_rssi_vec] + future_trajectory_vecs
        state = np.concatenate(state_parts)
        return torch.FloatTensor(state).unsqueeze(0)

    def generate_linear_trajectory(self, trajectory_len=TRAJECTORY_LEN):
        start_zone = np.random.randint(0, self.n_zones)
        trajectory = [start_zone]
        deltas = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        current_direction = random.choice(list(deltas.keys()))
        for _ in range(trajectory_len - 1):
            current_zone = trajectory[-1]
            x, y = current_zone // self.map_size, current_zone % self.map_size
            change_direction = random.random() < 0.15
            dx, dy = deltas[current_direction]
            next_x, next_y = x + dx, y + dy
            if not (0 <= next_x < self.map_size and 0 <= next_y < self.map_size):
                change_direction = True
            if change_direction:
                possible_directions = []
                for direction, (dx_p, dy_p) in deltas.items():
                    if 0 <= x + dx_p < self.map_size and 0 <= y + dy_p < self.map_size:
                        possible_directions.append(direction)
                current_direction = random.choice(possible_directions)
            dx_final, dy_final = deltas[current_direction]
            final_x, final_y = x + dx_final, y + dy_final
            next_zone = final_x * self.map_size + final_y
            trajectory.append(next_zone)
        return trajectory

    def reset(self, trajectory=None):
        if trajectory:
            self.trajectory = trajectory
        else:
            self.trajectory = self.generate_linear_trajectory()
        self.robot_zone_id = self.trajectory[0]
        self.traj_step = 0
        loads = self._get_current_ap_loads()
        best_ap, lowest_latency = -1, float('inf')
        
        initial_conditions = {} 
        
        for ap_id in range(1, self.n_aps + 1):
            rssi = self._calculate_rssi(self.robot_zone_id, ap_id)
            latency = self._calculate_latency(rssi, loads[ap_id])
            
            initial_conditions[ap_id] = {'rssi': rssi, 'latency': latency}
            
            if latency < lowest_latency:
                lowest_latency, best_ap = latency, ap_id
        self.current_ap = best_ap
        
        return self.get_state(), initial_conditions

    def get_valid_actions_mask(self):
        if self.traj_step >= len(self.trajectory) - 1:
            return torch.zeros(self.n_aps, dtype=torch.bool)

        next_zone = self.trajectory[self.traj_step + 1]
        valid_actions = torch.zeros(self.n_aps, dtype=torch.bool)
        for ap_id in range(1, self.n_aps + 1):
            rssi = self._calculate_rssi(next_zone, ap_id)
            if rssi >= self.RSSI_DISCONNECT_THRESHOLD:
                valid_actions[ap_id - 1] = True

        if not valid_actions.any():
            best_rssi, best_ap_idx = -float('inf'), -1
            for ap_id in range(1, self.n_aps + 1):
                 rssi = self._calculate_rssi(next_zone, ap_id)
                 if rssi > best_rssi:
                     best_rssi, best_ap_idx = rssi, ap_id - 1
            valid_actions[best_ap_idx] = True

        return valid_actions

    def step(self, action_ap):
        self.traj_step += 1
        next_zone = self.trajectory[self.traj_step]
        all_ap_stats = {}
        loads = self._get_current_ap_loads()
        for ap_id in range(1, self.n_aps + 1):
            rssi = self._calculate_rssi(next_zone, ap_id)
            latency = self._calculate_latency(rssi, loads[ap_id])
            all_ap_stats[ap_id] = {'rssi': rssi, 'latency': latency}
        cost = all_ap_stats[action_ap]['latency']
        if action_ap != self.current_ap:
            cost += self.T_SWITCH
        self.current_ap = action_ap
        self.robot_zone_id = next_zone
        reward = -cost
        done = (self.traj_step == len(self.trajectory) - 1)
        next_state = self.get_state() if not done else None
        return next_state, torch.tensor([reward], dtype=torch.float32), done, {'ap_stats': all_ap_stats}

    @staticmethod
    def render_trajectory_map(trajectory, map_size):
        grid = [[' . ' for _ in range(map_size)] for _ in range(map_size)]
        for zone in trajectory:
            row, col = zone // map_size, zone % map_size
            if grid[row][col] == ' . ': grid[row][col] = ' * '
        start_row, start_col = trajectory[0] // map_size, trajectory[0] % map_size
        end_row, end_col = trajectory[-1] // map_size, trajectory[-1] % map_size
        grid[start_row][start_col] = '[S]'
        grid[end_row][end_col] = '[E]'
        print("--- Trajectory Summary Map ---")
        border = f"+{'---' * map_size}+"
        map_str = f"AP 1 {border} AP 3\n"
        for r_idx, r in enumerate(grid): map_str += f"  |{''.join(r)}|\n"
        map_str += f"AP 2 {border} AP 4\n"
        map_str += "Legend: [S] = Start, [E] = End, * = Path\n"
        print(map_str)

# --- 2. Replay Buffer and DQN Agent ---
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done', 'next_valid_mask'))
class ReplayBuffer:
    def __init__(self, capacity): self.memory = deque([], maxlen=capacity)
    def push(self, *args): self.memory.append(Experience(*args))
    def sample(self, batch_size): return random.sample(self.memory, batch_size)
    def __len__(self): return len(self.memory)
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(state_size, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, action_size)
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.policy_net = QNetwork(state_size, action_size)
        self.target_net = QNetwork(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayBuffer(BUFFER_SIZE)
        self.steps_done = 0

    def select_action(self, state, valid_actions_mask):
        self.steps_done += 1
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        if random.random() > eps_threshold:
            with torch.no_grad():
                q_values = self.policy_net(state)
                q_values[0, ~valid_actions_mask] = -float('inf')
                return q_values.max(1)[1].view(1, 1)
        else:
            valid_indices = torch.where(valid_actions_mask)[0]
            action_idx = random.choice(valid_indices).item()
            return torch.tensor([[action_idx]], dtype=torch.long)

    def learn(self):
        if len(self.memory) < BATCH_SIZE: return
        experiences = self.memory.sample(BATCH_SIZE)
        batch = Experience(*zip(*experiences))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_valid_mask_batch = torch.stack(batch.next_valid_mask)
        non_final_mask = torch.tensor(tuple(s is not None for s in batch.next_state), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(BATCH_SIZE)
        if non_final_next_states.nelement() > 0:
            target_q_values = self.target_net(non_final_next_states)
            target_q_values[~next_valid_mask_batch[non_final_mask]] = -float('inf')
            next_state_values[non_final_mask] = target_q_values.max(1)[0].detach()
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

# --- 3. Smarter Greedy Agent Evaluation Function ---
def run_greedy_evaluation(env, evaluation_trajectories):
    total_costs = []
    for i, traj in enumerate(evaluation_trajectories):
        # NEW: Set a seed to ensure identical random conditions as the DQN agent for a fair comparison.
        np.random.seed(i)
        random.seed(i)

        _, initial_conditions = env.reset(trajectory=traj)
        done = False
        total_episode_cost = 0
        print(f"\n--- Greedy Episode {i+1} | Trajectory: {traj} ---")
        
        print(f"Initial Zone: {env.robot_zone_id}. Initial AP conditions:")
        stats_parts = [f"[AP{ap_id}: {s['rssi']:.1f}dBm, {s['latency']:.1f}ms]" for ap_id, s in initial_conditions.items()]
        print("  -> " + ", ".join(stats_parts))
        print(f"Step 0: Best initial choice is AP {env.current_ap} (Latency: {initial_conditions[env.current_ap]['latency']:.1f}ms)")
        
        while not done:
            prev_zone, prev_ap = env.robot_zone_id, env.current_ap
            valid_actions_mask = env.get_valid_actions_mask()
            valid_ap_indices = torch.where(valid_actions_mask)[0]
            if env.traj_step >= len(env.trajectory) - 1:
                break
            next_zone = env.trajectory[env.traj_step + 1]
            loads = env._get_current_ap_loads()
            potential_outcomes = []
            for ap_idx in valid_ap_indices:
                ap_id = ap_idx.item() + 1
                rssi = env._calculate_rssi(next_zone, ap_id)
                latency = env._calculate_latency(rssi, loads[ap_id])
                cost = latency + (0 if ap_id == prev_ap else env.T_SWITCH)
                potential_outcomes.append({'ap_id': ap_id, 'cost': cost})
            best_choice = min(potential_outcomes, key=lambda x: x['cost'])
            action_ap = best_choice['ap_id']
            _, reward, done, info = env.step(action_ap)
            step_cost = -reward.item()
            total_episode_cost += step_cost
            decision_log = f"SWITCH to AP {action_ap}" if action_ap != prev_ap else f"STAY with AP {action_ap}"
            print(f"Step {env.traj_step}: Z{prev_zone}->Z{env.robot_zone_id}. Decision: {decision_log}. Cost: {step_cost:.2f}")
            stats_log = "  -> AP Stats: "
            stats_parts = [f"[AP{ap_id}: {s['rssi']:.1f}dBm, {s['latency']:.1f}ms]" for ap_id, s in info['ap_stats'].items()]
            print(stats_log + ", ".join(stats_parts))
        print(f"--> Greedy Episode {i+1} Finished. Total Cost: {total_episode_cost:.2f}")
        env.render_trajectory_map(env.trajectory, env.map_size)
        total_costs.append(total_episode_cost)
    return np.mean(total_costs)

# --- 4. Main Execution Block ---
if __name__ == "__main__":
    env = WifiRLEnvironment()
    state_size = (env.n_zones +
                  env.n_aps +
                  env.n_aps +
                  env.n_aps +
                  env.lookahead_steps * env.n_zones)
    action_size = env.n_aps
    agent = DQNAgent(state_size, action_size)
    
    print(f"State size: {state_size}, Action size: {action_size}")
    print("="*60 + f"\nStarting Training for {N_EPISODES} episodes...\n" + "="*60)
    start_time = time.time()
    for i_episode in range(N_EPISODES):
        state, _ = env.reset()
        done = False
        while not done:
            valid_mask = env.get_valid_actions_mask()
            action_tensor = agent.select_action(state, valid_mask)
            action_ap = action_tensor.item() + 1
            next_state, reward, done, _ = env.step(action_ap)
            next_valid_mask = env.get_valid_actions_mask() if not done else torch.zeros(action_size, dtype=torch.bool)
            agent.memory.push(state, action_tensor, reward, next_state, done, next_valid_mask)
            state = next_state
            agent.learn()
        if i_episode % TARGET_UPDATE == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        if (i_episode + 1) % 100 == 0:
            print(f"Episode {i_episode+1}/{N_EPISODES} completed.")
    print(f"\n--- Training Finished in {time.time() - start_time:.2f} seconds ---\n")

    # --- Evaluation ---
    print("="*60 + "\nGenerating shared trajectories for a fair evaluation...\n" + "="*60)
    evaluation_trajectories = [env.generate_linear_trajectory() for _ in range(N_EVAL_EPISODES)]
    print("="*60 + "\n1. Evaluating Intelligent Agent (DQN)\n" + "="*60)
    dqn_costs = []
    for i, traj in enumerate(evaluation_trajectories):
        # NEW: Set a seed to ensure the random conditions can be replicated for the greedy agent.
        np.random.seed(i)
        random.seed(i)

        state, initial_conditions = env.reset(trajectory=traj)
        done = False
        total_cost = 0
        print(f"\n--- DQN Eval Episode {i+1} | Trajectory: {traj} ---")
        
        print(f"Initial Zone: {env.robot_zone_id}. Initial AP conditions:")
        stats_parts = [f"[AP{ap_id}: {s['rssi']:.1f}dBm, {s['latency']:.1f}ms]" for ap_id, s in initial_conditions.items()]
        print("  -> " + ", ".join(stats_parts))
        print(f"Step 0: Best initial choice is AP {env.current_ap} (Latency: {initial_conditions[env.current_ap]['latency']:.1f}ms)")

        while not done:
            prev_ap, prev_zone = env.current_ap, env.robot_zone_id
            valid_mask = env.get_valid_actions_mask()
            with torch.no_grad():
                q_values = agent.policy_net(state)
                q_values[0, ~valid_mask] = -float('inf')
                action_tensor = q_values.max(1)[1].view(1, 1)
            action_ap = action_tensor.item() + 1
            next_state, reward, done, info = env.step(action_ap)
            cost = -reward.item()
            total_cost += cost
            decision_log = f"SWITCH to AP {action_ap}" if action_ap != prev_ap else f"STAY with AP {action_ap}"
            print(f"Step {env.traj_step}: Z{prev_zone}->Z{env.robot_zone_id}. Decision: {decision_log}. Cost: {cost:.2f}")
            stats_log = "  -> AP Stats: "
            stats_parts = [f"[AP{ap_id}: {s['rssi']:.1f}dBm, {s['latency']:.1f}ms]" for ap_id, s in info['ap_stats'].items()]
            print(stats_log + ", ".join(stats_parts))
            state = next_state
        print(f"--> DQN Episode {i+1} Finished. Total Cost: {total_cost:.2f}")
        env.render_trajectory_map(env.trajectory, env.map_size)
        dqn_costs.append(total_cost)
    avg_dqn_cost = np.mean(dqn_costs)
    print("\n" + "="*60 + "\n2. Evaluating Baseline 'Smarter' Greedy Agent\n" + "="*60)
    avg_greedy_cost = run_greedy_evaluation(env, evaluation_trajectories)
    print("\n" + "="*60 + "\n3. Performance Comparison\n" + "="*60)
    print(f"Intelligent DQN Agent Average Cost....: {avg_dqn_cost:.2f} ms")
    print(f"Baseline Greedy Agent Average Cost...: {avg_greedy_cost:.2f} ms")
    if avg_greedy_cost > 0:
        improvement = ((avg_greedy_cost - avg_dqn_cost) / avg_greedy_cost) * 100
        if improvement > 0:
            print(f"\nThe DQN agent is {improvement:.2f}% more efficient than the greedy baseline.")
        else:
            print(f"\nThe DQN agent performed {-improvement:.2f}% worse.")
