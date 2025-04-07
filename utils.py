import torch
import gc
from psutil import virtual_memory

def free_gpu(verbose: bool = True):
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()
    if verbose:
        print("✅ GPU memory cache cleared.")

def gpu_info():
    if not torch.cuda.is_available():
        print("⚠️ CUDA unavailable.")
        return
    print(f"🖥 GPU: {torch.cuda.get_device_name(0)}")
    print(f"🚦 Memory used: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"📦 Reserved memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

def diagnostic_report(texts, labels=None, batch_size=None):
    print("")
    print("📋 === DIAGNOSTICS ===")
    print(f"🧾 Samples in dataset: {len(texts)}")
    if labels:
        print(f"🏷 Labels: {set(labels)}")

    avg_len = sum(len(t.split()) for t in texts) / len(texts)
    print(f"📏 Average text length (words): {avg_len:.1f}")

    bs = batch_size or (16 if torch.cuda.is_available() else 4)
    print(f"⚙️ batch_size: {bs}")

    gpu_info()

    ram = virtual_memory()
    print(f"🧠 RAM available: {ram.available // 1024**2} MB / Total: {ram.total // 1024**2} MB")
    print("✅ Diagnostics finished.")
    print("")