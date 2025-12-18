from abc import ABC, abstractmethod
from enum import Enum
import logging

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
        if beacons is None and self._state == RoamingState.CONNECTED:
            print("missed beacons")
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
        self._ap = ap
        self._env.process(self._roaming_process())

    def _disconnect(self) -> None:
        logger.info("Disconnected from AP {}".format(self._ap))
        self._state = RoamingState.DISCONNECTED
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
        self._rssi_list = None
        self._bad_beacons = 0
    
    def notify_beacon(self, pos, beacons):
        # check disconnectio due to missed beacons
        super().notify_beacon(pos, beacons)

        # update rssi list if beacon is received
        rssi_list = [binfo.rssi for binfo in beacons] if beacons else None
        if rssi_list is not None:
            self._rssi_list = rssi_list
        
        # if connected, if last three beacons were bad, roam
        if self._state == RoamingState.CONNECTED:
            if rssi_list is None or rssi_list[self._ap] < self._rssi_threshold:
                self._bad_beacons += 1
                if self._bad_beacons >= 3:
                    if self._rssi_list is not None:
                        print(rssi_list)
                        logger.info("Reached max bad beacons")
                        self._roam(self._get_best_ap())
                        self._bad_beacons = 0
            else:
                self._bad_beacons = 0

        # if disconnected, roam
        elif self._state == RoamingState.DISCONNECTED and self._rssi_list is not None:
            self._roam(self._get_best_ap())

    def notify_tx(self, pos, tx_info):
        pass

    def _get_best_ap(self):
        return self._rssi_list.index(max(self._rssi_list))
