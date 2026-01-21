import simpy
import random
import numpy as np
from roaming.utils import TupleRC, NetworkConfig, WifiParams, SimConfig
from roaming.plotter import MapPlotter
from roaming.roaming import DistanceRoaming, RSSIRoamingAlgorithm, OptimizedRoaming
from roaming.environment import SimpleWifiEnv, MapWifiEnv
from roaming.trajectory import TrajectorySimulator
from roaming.metrics import WifiMetric, WifiStat
import logging

logger = logging.getLogger(__name__)


DATA_FOLDER = "data"
CACHE_FOLDER = "cache"
SIMULATION_SEED = 2

MAP_DIMS = TupleRC(60, 120)
NET_CONFIG = NetworkConfig(
    net_name = "ns3_high_res_2xm0_3xm1",
    map_dims = MAP_DIMS,
    ap_positions = [
        TupleRC(0, 0),                              # Top-Left
        TupleRC(MAP_DIMS.row-1, 0),                 # Bottom-Left
        TupleRC(0, MAP_DIMS.col-1),                 # Top-Right
        TupleRC(MAP_DIMS.row-1, MAP_DIMS.col-1),    # Bottom-Right
        TupleRC(MAP_DIMS.row//2, MAP_DIMS.col//2)   # Center
    ],
    datasets = [("handover_hi_res", "map_0")]*2 + [("handover_hi_res", "map_1")]*3,
    ap_loads = [0.2, 0.2, 0.2, 0.2, 0.6]
)

WIFI_PARAMS = WifiParams (
    rssi_threshold = -80.0,
    roaming_time = 0.2,
    min_switch_time = 10
)

SIM_CONFIG = SimConfig(
    trajectory_len = 50,
    pkt_period = 0.1,
    speed = 0.5,
    beacon_time = 1.0,
    tx_start_time = 2.0,
)

def main():
    # Simulation seed for reproducibility
    random.seed(SIMULATION_SEED)
    np.random.seed(SIMULATION_SEED)

    # Create simpy environment
    env = simpy.Environment()

    # Enable logging
    logging.basicConfig(level=logging.INFO)

    # Create wifi environment
    # wifi_env = SimpleWifiEnv(net_conf=NET_CONFIG, wifi_params=WIFI_PARAMS)
    wifi_env = MapWifiEnv(net_conf=NET_CONFIG, data_dir=DATA_FOLDER, cache_dir=CACHE_FOLDER, seed=SIMULATION_SEED)
    wifi_env.load_datasets(datasets=NET_CONFIG.datasets)

    # Create roaming algorithm
    # roam_alg = DistanceRoaming(env=env, wifi_sim=wifi_env, roaming_time=0.2)
    # roam_alg = RSSIRoamingAlgorithm(env=env, wifi_sim=wifi_env, roaming_time=0.2, rssi_threshold=-80)
    roam_alg = OptimizedRoaming(env, wifi_sim=wifi_env, roaming_time=WIFI_PARAMS.roaming_time, metric=WifiMetric.NUM_TRIES, stat=WifiStat.MEAN, min_switch_time=WIFI_PARAMS.min_switch_time)

    # Create trajectory simulator
    traj_sim = TrajectorySimulator(env=env, wifi_sim=wifi_env, roam_alg=roam_alg, sim_config=SIM_CONFIG, cache_dir=CACHE_FOLDER, net_name=NET_CONFIG.net_name)

    # Configure roaming algorithm (only for OptimalRoaming)
    roam_alg.configure(traj_sim)

    # Run simulation
    logging.info("Starting simulation")
    env.run()
    logging.info("Simulation Ended")

    # # Trajectory plots
    map_plt = MapPlotter(wifi_sim=wifi_env, cache_dir=CACHE_FOLDER, net_name=NET_CONFIG.net_name)
    map_plt.generate_maps()
    map_plt.plot_maps(traj_num=traj_sim.traj_num, interactive=False)


if __name__ == "__main__":
    main()

# TODO
# [done] fix structure of cache folder
# [done] change exp_name to net_name
# [done] save plots of trajectories
# [done] sfasare beacons rispetto ai messaggi (random offset)
# [done] RSSI roaming with 1s scanning and multiple thresholds
# [done] start transmissions after offset
# [done] add possibility of saving multiple trajectories and plots
# save trajectory configuration
# load configuration from file
# add type hints to functions
# fix logging messages

# optimal roaming fixes/improvements
# - [done] updates on trajectory should be made every beacon time
# - [done] specify min, max for the optimization metric
# - [done] avoid too many switches by removing short roaming intervals
# - check behavior on disconnections

# [OPTIONAL]
# implement packet queuing when when you roam or disconnect (for more realistic latency) -> useful also if optimizing another metric

# MEETING
# pallini più grandi
# - adattatore principale che sa RSSI
# - interfaccia ausialira che fa scanning
# - ricerca si fa su 5 canali
# - 5 canali, 1 secondo (può essere ridotto) -> Per RSSI (tempo di scansione)
# - valuto distanza ogni 0.1 secondo) -> tempo determinazione posizione (dipende dalla velocità del nodo)
# - chiedere articolo altri organizzatori

