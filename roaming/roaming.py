from abc import ABC, abstractmethod
from enum import Enum
import logging

from roaming.utils import TupleRC
from roaming.metrics import WifiMetric, WifiStat, get_comf_funct

logger = logging.getLogger(__name__)


class RoamingState(Enum):
    DISCONNECTED = 1
    CONNECTED = 2
    ROAMING = 3


class RoamingAlgorithm(ABC):
    def __init__(self, env, wifi_sim, roaming_time):
        self._env = env
        self._wifi_sim = wifi_sim
        self._roaming_time = roaming_time
        self._ap = None
        self._state = RoamingState.DISCONNECTED
        self._missed_beacons = 0
        self._count = 0

    @property
    def state(self) -> RoamingState:
        return self._state

    @property
    def connected(self) -> bool:
        return self._state == RoamingState.CONNECTED

    @property
    def ap(self) -> int:
        return self._ap
    
    @property
    def count(self) -> int:
        return self._count
    
    def notify_beacon(self, pos, beacons) -> None:
        # disconnection after three missed beacons
        if self._state == RoamingState.CONNECTED and beacons[self._ap] is None:
            logging.info("Missed beacons")
            self._missed_beacons += 1
            if self._missed_beacons >= 3:
                self._disconnect()
        else:
            self._missed_beacons = 0

    @abstractmethod
    def notify_tx(self, pos, tx_info) -> None:
        pass

    def _roaming_process(self):
        self._state = RoamingState.ROAMING
        self._count += 1
        logger.info("Roaming to AP {}".format(self._ap))
        yield self._env.timeout(self._roaming_time)
        self._state = RoamingState.CONNECTED
        logger.info("Connected to to AP {}".format(self._ap))

    def _roam(self, ap) -> None:
        self._ap = ap# Create wifi environment
    # wifi_env = SimpleWifiEnv(net_conf=NET_CONFIG, wifi_params=WIFI_PARAMS)
        self._env.process(self._roaming_process())

    def _disconnect(self) -> None:
        logger.info("Disconnected from AP {}".format(self._ap))
        self._state = RoamingState.DISCONNECTED
        self._ap = None
        self._missed_beacons = 0


class DistanceRoaming(RoamingAlgorithm):
    def __init__(self, env, wifi_sim, roaming_time):
        super().__init__(env, wifi_sim, roaming_time)

    def notify_beacon(self, pos, beacons):
        # check disconnectio due to missed beacons
        super().notify_beacon(pos, beacons)
        # if not roaming, connect/switch to best AP
        if not self._state == RoamingState.ROAMING:
            ap_dist = [(pos - ap_pos).norm() for ap_pos in self._wifi_sim.ap_positions]
            best_ap = ap_dist.index(min(ap_dist))
            if self._state == RoamingState.DISCONNECTED or best_ap != self._ap:
                self._roam(best_ap)

    def notify_tx(self, pos, tx_info):
        pass


class RSSIRoamingAlgorithm(RoamingAlgorithm):
    def __init__(self, env, wifi_sim, roaming_time, rssi_threshold):
        super().__init__(env, wifi_sim, roaming_time)
        self._rssi_threshold = rssi_threshold
        self._rssi_list = [None]*self._wifi_sim.n_aps
        self._bad_beacons = 0
    
    def notify_beacon(self, pos, beacons):
        # check disconnectio due to missed beacons
        super().notify_beacon(pos, beacons)

        # update rssi list if beacon is received
        self._rssi_list = [binfo.rssi if binfo is not None else None for binfo in beacons]
        
        # if connected, if last three beacons were bad, roam
        if self._state == RoamingState.CONNECTED:
            if self._rssi_list[self._ap] is None or self._rssi_list[self._ap] < self._rssi_threshold:
                self._bad_beacons += 1
                if self._bad_beacons >= 3:
                    if any([rssi is not None for rssi in self._rssi_list]):
                        logger.info("Reached max bad beacons")
                        best_ap = self._get_best_ap()
                        if best_ap != self._ap:
                            self._roam(best_ap)
                            self._bad_beacons = 0
            else:
                self._bad_beacons = 0

        # if disconnected, roam
        elif self._state == RoamingState.DISCONNECTED and self._rssi_list is not None:
            self._roam(self._get_best_ap())

    def notify_tx(self, pos, tx_info):
        pass

    def _get_best_ap(self):
        return self._rssi_list.index(max([rssi for rssi in self._rssi_list if rssi is not None]))


class OptimizedRoaming(RoamingAlgorithm):
    def __init__(self, env, wifi_sim, roaming_time, metric: WifiMetric, stat: WifiStat, min_switch_time=None):
        super().__init__(env, wifi_sim, roaming_time)
        self._metric = metric
        self._stat = stat
        self._comp_f = get_comf_funct(self._metric)
        # trajectory info        
        self._traj_sim = None        
        self._step = None
        self._segment = None
        # swithing info
        self._switch_points = None
        self._switch_aps = None
        self._min_switch_time = min_switch_time

    def configure(self, traj_sim):
        self._traj_sim = traj_sim
        self._step = (self._traj_sim.sim_config.speed * self._traj_sim.sim_config.beacon_time)
        self._traj_sim.register_traj_change_callback(lambda segment, pos: self._traj_change_callback(segment, pos))

    def notify_tx(self, pos, tx_info) -> None:
        pass

    def notify_beacon(self, pos, beacons):
        super().notify_beacon(pos, beacons)
        if self.state != RoamingState.ROAMING:
                best_ap = self._best_ap(pos)
                if best_ap is not None:
                    self._roam(best_ap)
    
    def _traj_change_callback(self, pos: TupleRC, segment: tuple[TupleRC, TupleRC]):
        self._segment = segment
        
        len_seg =(self._segment[1] - self._segment[0]).norm()
        num_samples = round(len_seg / (self._step / 2))
        step_vect = (self._segment[1] - self._segment[0]) / num_samples
        positions = [self._segment[0] + step_vect * (i+1) for i in range(num_samples)]
        
        # find get metrics and best ap in each trajectory point
        best_aps = []
        pos_metrics = []
        for p in positions:
            metrics = [self._wifi_sim.get_metrics(p, ap) for ap in range(self._wifi_sim.n_aps)]
            metrics = [m[self._metric][self._stat] if m is not None else None for m in metrics]
            best_ap = metrics.index(self._comp_f([m for m in metrics if m is not None]))
            best_aps.append(best_ap)
            pos_metrics.append(metrics)

        # prev_switch_time = None
        # if self._ap is not None:
        #     if self._switch_aps and self._switch_aps[-1] == self._ap:
        #         prev_switch_time = ((self._switch_points[-1] - self._segment[0]).norm()) / self._traj_sim.sim_config.speed

        # find switch points and switch APs
        best_aps = [self._ap] + best_aps
        switch_idxs = [i for i in range(len(best_aps) -1) if best_aps[i] != best_aps[i+1]]
        self._switch_points = [positions[i] for i in switch_idxs]
        self._switch_aps = [best_aps[i+1] for i in switch_idxs]

        # remove short switching intervals
        if self._min_switch_time is not None:
            idx_interval = 0
            switch_times = [(self._switch_points[i+1] - self._switch_points[i]).norm() / self._traj_sim.sim_config.speed for i in range(len(self._switch_points) -1)]
            while idx_interval < len(switch_times):
                if switch_times[idx_interval] < self._min_switch_time and not (idx_interval == 0 and self._ap == None):
                    ap_prev = self._switch_aps[idx_interval -1] if idx_interval > 0 else self._ap
                    ap_next = self._switch_aps[idx_interval + 1]
                    pos_idx_prev = switch_idxs[idx_interval]
                    pos_idx_next = switch_idxs[idx_interval + 1]
                    metrics_prev = [pos_metrics[i][ap_prev] for i in range(pos_idx_prev, pos_idx_next)]
                    metrics_next = [pos_metrics[i][ap_next] for i in range(pos_idx_prev, pos_idx_next)]
                    avg_prev = sum(metrics_prev) / len(metrics_prev) if not any([m is None for m in metrics_prev]) else None
                    avg_next = sum(metrics_next) / len(metrics_next) if not any([m is None for m in metrics_next]) else None

                    if avg_prev is None and avg_next is None:
                        idx_interval+=1
                        continue
                    if avg_prev is not None and (avg_next is None or avg_prev >= avg_next):
                        self._switch_points.pop(idx_interval)
                        self._switch_aps.pop(idx_interval)
                        switch_idxs.pop(idx_interval)                        
                    else:
                        self._switch_aps[idx_interval] = ap_next
                        self._switch_points.pop(idx_interval+1)
                        self._switch_aps.pop(idx_interval+1)
                        switch_idxs.pop(idx_interval+1)

                    # update structures
                    switch_times = [(self._switch_points[i+1] - self._switch_points[i]).norm() / self._traj_sim.sim_config.speed for i in range(len(self._switch_points) -1)]
                else:
                    idx_interval+=1        

        # fix problem of first use / use when changing segment
        if self._state != RoamingState.ROAMING:
            best_ap = self._best_ap(pos)
            if best_ap is not None:
                self._roam(best_ap)
                logger.info("roaming")

    def _best_ap(self, pos) -> int | None:
        dist_pos = (pos - self._segment[0]).norm()
        dist_switch = [(switch_point - self._segment[0]).norm() for switch_point in self._switch_points]
        dist_switch = [(idx, dist) for idx, dist in enumerate(dist_switch) if dist>=dist_pos and dist_pos+self._step >= dist and self._switch_aps[idx] != self._ap]
        if dist_switch:
            idx, _ = dist_switch[0]
            return self._switch_aps[idx]
        return None
    