import simpy
import random
import numpy as np
import logging
from pathlib import Path
from roaming.utils import TupleRC, NetworkConfig, WifiConfig, SimConfig, ExpConfig, Config, load_config, save_config
from roaming.plotter import MapPlotter
from roaming.roaming import DistanceRoaming, RSSIRoaming, OptimizedRoaming
from roaming.environment import SimpleWifiEnv, MapWifiEnv
from roaming.trajectory import TrajectorySimulator
from roaming.metrics import WifiMetric, WifiStat

logger = logging.getLogger(__name__)


EXP_CONF = ExpConfig(
    data_dir = "data",
    cache_dir = "cache",
    sim_seed = 2,
    exp_name = "exp_plot"
)

MAP_DIMS = TupleRC(101, 101)
NET_CONF = NetworkConfig(
    map_dims = MAP_DIMS,
    ap_positions = [
        TupleRC(50, 50),                              # Top-Left
        # TupleRC(MAP_DIMS.row-1, 0),                 # Bottom-Left
        # TupleRC(0, MAP_DIMS.col-1),                 # Top-Right
        # TupleRC(MAP_DIMS.row-1, MAP_DIMS.col-1),    # Bottom-Right
        # TupleRC(MAP_DIMS.row//2, MAP_DIMS.col//2)   # Center
    ],
    datasets = [("handover_hi_res", "map_0")], #*2 + [("handover_hi_res", "map_1")]*3,
    ap_loads = [0.2] #, 0.2, 0.2, 0.2, 0.6]
)

# MAP_DIMS = TupleRC(60, 120)
# NET_CONF = NetworkConfig(
#     map_dims = MAP_DIMS,
#     ap_positions = [
#         TupleRC(0, 0),                              # Top-Left
#         TupleRC(MAP_DIMS.row-1, 0),                 # Bottom-Left
#         TupleRC(0, MAP_DIMS.col-1),                 # Top-Right
#         TupleRC(MAP_DIMS.row-1, MAP_DIMS.col-1),    # Bottom-Right
#         TupleRC(MAP_DIMS.row//2, MAP_DIMS.col//2)   # Center
#     ],
#     datasets = [("handover_hi_res", "map_0")]*2 + [("handover_hi_res", "map_1")]*3,
#     ap_loads = [0.2, 0.2, 0.2, 0.2, 0.6]
# )

WIFI_CONF = WifiConfig (
    roaming_alg = "OptimizedRoaming",
    rssi_threshold = -75.0,
    roaming_time = 0.2,
    min_switch_time = 0.2
)

SIM_CONF = SimConfig(
    trajectory_len = 1500,
    pkt_period = 0.1,
    speed = 0.5,
    beacon_time = 0.1,
    tx_start_time = 2.0,
)


def main():
    # Simulation seed for reproducibility
    random.seed(EXP_CONF.sim_seed)
    np.random.seed(EXP_CONF.sim_seed)

    # Create simpy environment
    # env = simpy.Environment()

    # Create wifi environment
    # wifi_env = SimpleWifiEnv(net_conf=NET_CONFIG, wifi_params=WIFI_CONFIG)
    wifi_env = MapWifiEnv(net_conf=NET_CONF, data_dir=EXP_CONF.data_dir, cache_dir=EXP_CONF.cache_dir, seed=EXP_CONF.sim_seed)
    wifi_env.load_datasets(datasets=NET_CONF.datasets)

    # # Create roaming algorithm
    # roam_alg = None
    # match WIFI_CONF.roaming_alg:
    #     case "RSSIRoaming":
    #         roam_alg = RSSIRoaming(env=env, wifi_sim=wifi_env, roaming_time=WIFI_CONF.roaming_time, rssi_threshold=WIFI_CONF.rssi_threshold)
    #     case "DistanceRoaming":
    #         roam_alg = DistanceRoaming(env=env, wifi_sim=wifi_env, roaming_time=WIFI_CONF.roaming_time)
    #     case "OptimizedRoaming":
    #         roam_alg = OptimizedRoaming(env, wifi_sim=wifi_env, roaming_time=WIFI_CONF.roaming_time, metric=WifiMetric.NUM_TRIES, stat=WifiStat.MEAN, min_switch_time=WIFI_CONF.min_switch_time)
    #     case _:
    #         raise RuntimeError("Invalid algorithm algorithm: {}".format(WIFI_CONF.roaming_alg))

    # # Create trajectory simulator
    # traj_sim = TrajectorySimulator(env=env, wifi_sim=wifi_env, roam_alg=roam_alg, sim_config=SIM_CONF, cache_dir=EXP_CONF.cache_dir, exp_name=EXP_CONF.exp_name)

    # # Configure roaming algorithm (only for OptimalRoaming)
    # if WIFI_CONF.roaming_alg == "OptimizedRoaming":
    #     roam_alg.configure(traj_sim)

    # # Setup logger
    # log_folder = Path(EXP_CONF.cache_dir) / "experiments" / EXP_CONF.exp_name / "logs"
    # log_folder.mkdir(parents=True, exist_ok=True)
    # log_file = log_folder/"{:02d}.log".format(traj_sim.traj_num)
    # logging.basicConfig(filename=log_file, filemode="w", level=logging.INFO)

    # # Run simulation
    # logger.info("Starting simulation")
    # env.run()
    # logger.info("Simulation Ended")

    # # Save configuration
    # save_config(Config(exp_conf=EXP_CONF, net_conf=NET_CONF, wifi_conf=WIFI_CONF, sim_conf=SIM_CONF), traj_num=traj_sim.traj_num)

    # Trajectory plots
    map_plt = MapPlotter(wifi_sim=wifi_env, cache_dir=EXP_CONF.cache_dir, exp_name=EXP_CONF.exp_name)
    map_plt.generate_maps()
    map_plt.plot_maps(traj_num=None, interactive=False, extension="pdf")


if __name__ == "__main__":
    main()

# TODO
# [done] fix structure of cache folder
# [done] change exp_name to exp_name
# [done] save plots of trajectories
# [done] sfasare beacons rispetto ai messaggi (random offset)
# [done] RSSI roaming with 1s scanning and multiple thresholds
# [done] start transmissions after offset
# [done] add possibility of saving multiple trajectories and plots
# [done] save experiment configuration
# load configuration from file
# add type hints to functions
# make plot colors more visible
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

