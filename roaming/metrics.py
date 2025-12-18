import statistics
from enum import Enum, IntEnum
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

class DsField(IntEnum):
    seq = 0
    acked = 1
    latency = 2
    transmissions = 3
    rssi = 4
    noise = 5



def remove_dropped(data):
    data = data[data[DsField.acked.name] == True]
    if data.empty:
        raise NoDataException
    return data

METRICS_FN = {
    WifiMetric.LATENCY: {
        WifiStat.MEAN: lambda data: np.mean(remove_dropped(data)[DsField.latency.name].values),
        WifiStat.MIN: lambda data: np.min(remove_dropped(data)[DsField.latency.name].values),
        WifiStat.PERC_99: lambda data: np.percentile(remove_dropped(data)[DsField.latency.name].values, 99),
        WifiStat.PERC_99_9: lambda data: np.percentile(remove_dropped(data)[DsField.latency.name].values, 99.9)
    },
    WifiMetric.NUM_TRIES : {
        WifiStat.MEAN : lambda data: np.mean(remove_dropped(data)[DsField.transmissions.name].values),
        WifiStat.PERC_99 : lambda data: np.percentile(remove_dropped(data)[DsField.transmissions.name].values, 99),
        WifiStat.PERC_99_9 : lambda data: np.percentile(remove_dropped(data)[DsField.transmissions.name].values, 99.9)
    },
    WifiMetric.PLR: {
        WifiStat.NONE: lambda data: data[data[DsField.acked.name] == False].shape[0] / data.shape[0] * 100
    },
    WifiMetric.RSSI: {
        WifiStat.MEAN: lambda data: np.mean(data[DsField.rssi.name].values),
    },
    WifiMetric.SNR: {
        WifiStat.MEAN: lambda data: np.mean((data[DsField.rssi.name] - data[DsField.noise.name]).values),
    }
}