import psutil
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetCount, nvmlDeviceGetUtilizationRates

nvmlInit()
gpu_count = nvmlDeviceGetCount()
handles = [nvmlDeviceGetHandleByIndex(i) for i in range(gpu_count)]

def get_load():
    cpu = psutil.cpu_percent(interval=None)
    gpu = [nvmlDeviceGetUtilizationRates(h).gpu for h in handles]
    return cpu, gpu
