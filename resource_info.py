import time
import pynvml
import psutil

# 初始化 NVML
pynvml.nvmlInit()

# 获取 GPU 设备数量
device_count = pynvml.nvmlDeviceGetCount()

while True:
    try:
        for i in range(device_count):
            # 获取第 i 个 GPU 设备的句柄
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            # 获取 GPU 的利用率信息
            utilization_info = pynvml.nvmlDeviceGetUtilizationRates(handle)

            # 获取 GPU 的显存信息
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            # 打印 GPU 的编号、利用率和显存使用情况
            gpu_info = f"GPU {i}: {utilization_info.gpu}%, " + f"Memory Used = {memory_info.used / (1024 ** 2):.2f}" + f"/{memory_info.total / (1024 ** 2):.2f} MB|"
            # 获取当前使用 GPU 的进程信息
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            for process in processes:
                try:
                    pid = process.pid
                    process_info = psutil.Process(pid)
                    memory_percent = process_info.memory_percent()
                    cpu_percent = process_info.cpu_percent(interval=0.1)
                    print(gpu_info+f" CPU Memory Usage = {memory_percent:.2f}%, CPU {cpu_percent:.2f}% ")
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue

        # 每秒监控一次
        time.sleep(0.2)
    except Exception as e:
        print(f"An error occurred: {e}. Continuing monitoring...")
        time.sleep(1)
        continue
    except KeyboardInterrupt:
        print("Monitoring stopped by user.")
        break

# 关闭 NVML
pynvml.nvmlShutdown()