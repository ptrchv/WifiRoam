import statistics
from enum import Enum
import numpy as np

class NoDataException(Exception):
    pass

class WifiMetric(Enum):
    RSSI = 1
    SNR = 2
    LATENCY = 3
    NUM_TRIES = 4
    PLR = 5

class WifiStat(Enum):    
    MEAN = 1
    MIN = 2
    PERC_99 = 3
    PERC_99_9 = 4
    NONE = 5

def remove_dropped(data):
    data = [row for row in data if row["acked"] == True]
    if not data:
        raise NoDataException
    return data


METRICS_INFO = {
    WifiMetric.LATENCY: {
        "label": "Latency ($\mu$s)",
        "scaling": 0.001,
    },
    WifiMetric.NUM_TRIES: {
        "label": "# Transmissions",
        "scaling": 1,
    },
    WifiMetric.PLR: {
        "label": "PLR (%)",
        "scaling": 1,
    },
    WifiMetric.RSSI: {
        "label": "dbm",
        "scaling": 1,
    },
    WifiMetric.SNR: {
        "label": "dbm",
        "scaling": 1,
    }
}

def remove_dropped(data):
    data = data[data["acked"] == True]
    if data.empty:
        raise NoDataException
    return data

METRICS_FN = {
    WifiMetric.LATENCY: {
        WifiStat.MEAN: lambda data: np.mean(remove_dropped(data)["latency"].values),
        WifiStat.MIN: lambda data: np.min(remove_dropped(data)["latency"].values),
        WifiStat.PERC_99: lambda data: np.percentile(remove_dropped(data)["latency"].values, 99),
        WifiStat.PERC_99_9: lambda data: np.percentile(remove_dropped(data)["latency"].values, 99.9)
    },
    WifiMetric.NUM_TRIES : {
        WifiStat.MEAN : lambda data: np.mean(remove_dropped(data)["transmissions"].values),
        WifiStat.PERC_99 : lambda data: np.percentile(remove_dropped(data)["transmissions"].values, 99),
        WifiStat.PERC_99_9 : lambda data: np.percentile(remove_dropped(data)["transmissions"].values, 99.9)
    },
    WifiMetric.PLR: {
        WifiStat.NONE: lambda data: data[data["acked"] == False].shape[0] / data.shape[0] * 100
    },
    WifiMetric.RSSI: {
        WifiStat.MEAN: lambda data: np.mean(data["rssi"].values),
    },
    WifiMetric.SNR: {
        WifiStat.MEAN: lambda data: np.mean((data["rssi"] - data["noise"]).values),
    }
}