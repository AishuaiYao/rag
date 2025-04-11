import time
import pynvml

# 初始化 NVML
pynvml.nvmlInit()

# 获取 GPU 设备数量
device_count = pynvml.nvmlDeviceGetCount()

try:
    while True:
        for i in range(device_count):
            # 获取第 i 个 GPU 设备的句柄
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            # 获取 GPU 的利用率信息
            info = pynvml.nvmlDeviceGetUtilizationRates(handle)
            # 打印 GPU 的编号和利用率
            print(f"GPU {i}: Utilization = {info.gpu}%")
        # 每秒监控一次
        time.sleep(1)

except KeyboardInterrupt:
    print("Monitoring stopped by user.")
finally:
    # 关闭 NVML
    pynvml.nvmlShutdown()
