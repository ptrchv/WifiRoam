class RoamingAlgorithm:
    def __init__(self, rssi_threshold, roaming_time, eval_window):
        self._rssi_threshold = rssi_threshold
        self._roaming_time = roaming_time
        self._eval_window = eval_window
        self._prev_bad_rssi = None
        self._ap = None

    def update_state(self, time, rssi):
        if rssi < self._rssi_threshold:
            if self._prev_bad_rssi is None:
                self._prev_bad_rssi = time
            else:
                if time - self._prev_bad_rssi >= self._eval_window:
                    self._roam = True
                    self._prev_bad_rssi = None
        else:
            self._prev_bad_rssi = None

    def get_best_ap(self, rssi_samples):
        best_ap, best_rssi = None, None
        for ap, rssi in enumerate(rssi_samples):
            if best_rssi is None or rssi > best_rssi:
                best_ap, best_rssi = ap, rssi
        self._ap = best_ap
    
    @property
    def roaming(self):
        return True
    
    def ap(self):
        return self._ap
