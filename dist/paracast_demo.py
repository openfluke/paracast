# paracast_quick_demo_no_gpu.py
# run from: dist/   ->   python3 paracast_quick_demo_no_gpu.py
import os, ctypes, random, time


HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

RTLD_GLOBAL = getattr(ctypes, "RTLD_GLOBAL", 0)
lib = ctypes.CDLL(os.path.join(HERE, "libparacast.so"), mode=RTLD_GLOBAL)

# ctypes aliases
c_i, c_f, c_ll, c_p = ctypes.c_int, ctypes.c_float, ctypes.c_longlong, ctypes.c_void_p

# ---- declare C signatures (adapt these if your ABI differs) ----
lib.Paracast_CreateLargeNetwork.restype = c_ll

lib.Paracast_DisableGPU.argtypes = [c_ll]
lib.Paracast_DisableGPU.restype  = None

lib.Paracast_GenerateTestData.argtypes = [ctypes.POINTER(c_f), c_i, c_i]
lib.Paracast_GenerateTestData.restype  = None

lib.Paracast_Forward.argtypes = [c_ll, ctypes.POINTER(c_f), c_i, c_i]
lib.Paracast_Forward.restype  = c_i

lib.Paracast_GetOutput.argtypes = [c_ll, ctypes.POINTER(c_f), c_i]
lib.Paracast_GetOutput.restype  = c_i

lib.Paracast_ForwardBatch.argtypes = [c_ll, ctypes.POINTER(c_f), c_i, ctypes.POINTER(c_f)]
lib.Paracast_ForwardBatch.restype  = c_ll  # ns

lib.Paracast_Train.argtypes = [
    c_ll,
    ctypes.POINTER(c_f), c_i, c_i,    # inputs, inW, inH
    ctypes.POINTER(c_f), c_i, c_i,    # targets, outW, outH
    c_i,                              # batch
    c_i,                              # epochs
    c_f,                              # lr
    c_i,                              # useGPU (0/1)
    c_f, c_f                          # clipLower, clipUpper
]
lib.Paracast_Train.restype = c_i

lib.Paracast_GetLastError.restype = ctypes.c_char_p

lib.Paracast_Free.argtypes = [c_ll]
lib.Paracast_Free.restype  = None

# ---- small helpers ----
def err(where: str):
    msg = lib.Paracast_GetLastError()
    raise RuntimeError(f"{where} failed: {(msg or b'?').decode()}")

def check(ok: int, where: str):
    if not ok: err(where)

def fmt(vec, n=10):
    return "[" + ", ".join(f"{v:.3f}" for v in vec[:n]) + (", ..." if len(vec) > n else "") + "]"

def throughput(samples: int, ns: int) -> float:
    return samples / (ns / 1e9) if ns > 0 else float("nan")

# ---- demo params (adjustable) ----
IN_W, IN_H   = 28, 28
OUT_W, OUT_H = 10, 1
OUT_SIZE     = OUT_W * OUT_H

SEED_SAMPLE_NOISE = 0.02
NUDGE_CLASS_1     = 9  # First training: nudge to class 9
NUDGE_CLASS_2     = 0  # Second training: nudge to class 0 for wild change
EPOCHS_TRAIN_1    = 10 # Super short for quick demo
EPOCHS_TRAIN_2    = 10
LR_TRAIN          = 0.1 # Higher LR for wilder shifts (even with few epochs)

BATCH_FOR_STRESS  = 4096  # large batch for stress test
STRESS_ROUNDS     = 3     # a few heavy passes

def main():
    print("🔧 create model")
    net = lib.Paracast_CreateLargeNetwork()
    if net == 0: err("Paracast_CreateLargeNetwork")

    try:
        # single deterministic input
        random.seed(42)
        one_in = (c_f * (IN_W * IN_H))()
        for i in range(IN_W * IN_H):
            one_in[i] = random.random()

        # forward BEFORE any training
        print("\n▶️  forward (cpu) — before any training")
        start_time = time.perf_counter()
        check(lib.Paracast_Forward(net, one_in, IN_W, IN_H), "Paracast_Forward(before)")
        fwd_time = time.perf_counter() - start_time
        out_buf = (c_f * OUT_SIZE)()
        check(lib.Paracast_GetOutput(net, out_buf, OUT_SIZE) > 0, "Paracast_GetOutput(before)")
        out_before = [float(out_buf[i]) for i in range(OUT_SIZE)]
        argmax_before = max(range(OUT_SIZE), key=lambda j: out_before[j])
        print(f"forward time: {fwd_time:.4f}s")
        print("output (first 10):", fmt(out_before), f" | argmax = {argmax_before}")

        # make a training batch that nudges towards NUDGE_CLASS_1
        B = 128
        batch_x = (c_f * (B * IN_W * IN_H))()
        batch_y = (c_f * (B * OUT_SIZE))()

        for b in range(B):
            for i in range(IN_W * IN_H):
                batch_x[b*IN_W*IN_H + i] = one_in[i] + SEED_SAMPLE_NOISE * (random.random() - 0.5)

        for b in range(B):
            base = b * OUT_SIZE
            for j in range(OUT_SIZE):
                batch_y[base + j] = 1.0 if j == NUDGE_CLASS_1 else 0.0

        # First training round (CPU only)
        print(f"\n🎓 first train (cpu) — nudging to class {NUDGE_CLASS_1}, {EPOCHS_TRAIN_1} epochs, LR={LR_TRAIN}")
        start_time = time.perf_counter()
        ok = lib.Paracast_Train(
            net,
            batch_x, IN_W, IN_H,
            batch_y, OUT_W, OUT_H,
            B,
            EPOCHS_TRAIN_1,
            c_f(LR_TRAIN),
            0,  # CPU only
            c_f(-1e9), c_f(1e9)
        )
        train_time_1 = time.perf_counter() - start_time
        check(ok, "Paracast_Train(first)")
        print(f"train time: {train_time_1:.4f}s")

        # forward AFTER first training
        print("▶️  forward (cpu) — after first train")
        start_time = time.perf_counter()
        check(lib.Paracast_Forward(net, one_in, IN_W, IN_H), "Paracast_Forward(after first)")
        fwd_time_after1 = time.perf_counter() - start_time
        check(lib.Paracast_GetOutput(net, out_buf, OUT_SIZE) > 0, "Paracast_GetOutput(after first)")
        out_after_1 = [float(out_buf[i]) for i in range(OUT_SIZE)]
        argmax_after1 = max(range(OUT_SIZE), key=lambda j: out_after_1[j])
        print(f"forward time: {fwd_time_after1:.4f}s")
        print("output (first 10):", fmt(out_after_1), f" | argmax = {argmax_after1}")
        print(f"Change in argmax: {argmax_before} -> {argmax_after1}")

        # Second training: reuse the same batch but nudge to different class for wild change
        for b in range(B):
            base = b * OUT_SIZE
            for j in range(OUT_SIZE):
                batch_y[base + j] = 1.0 if j == NUDGE_CLASS_2 else 0.0

        # Second training round (CPU only)
        print(f"\n🎓 second train (cpu) — now nudging to class {NUDGE_CLASS_2}, {EPOCHS_TRAIN_2} epochs, LR={LR_TRAIN}")
        start_time = time.perf_counter()
        ok = lib.Paracast_Train(
            net,
            batch_x, IN_W, IN_H,
            batch_y, OUT_W, OUT_H,
            B,
            EPOCHS_TRAIN_2,
            c_f(LR_TRAIN),
            0,  # CPU only
            c_f(-1e9), c_f(1e9)
        )
        train_time_2 = time.perf_counter() - start_time
        check(ok, "Paracast_Train(second)")
        print(f"train time: {train_time_2:.4f}s")

        # forward AFTER second training
        print("▶️  forward (cpu) — after second train")
        start_time = time.perf_counter()
        check(lib.Paracast_Forward(net, one_in, IN_W, IN_H), "Paracast_Forward(after second)")
        fwd_time_after2 = time.perf_counter() - start_time
        check(lib.Paracast_GetOutput(net, out_buf, OUT_SIZE) > 0, "Paracast_GetOutput(after second)")
        out_after_2 = [float(out_buf[i]) for i in range(OUT_SIZE)]
        argmax_after2 = max(range(OUT_SIZE), key=lambda j: out_after_2[j])
        print(f"forward time: {fwd_time_after2:.4f}s")
        print("output (first 10):", fmt(out_after_2), f" | argmax = {argmax_after2}")
        print(f"Change in argmax: {argmax_after1} -> {argmax_after2}")

        print(f"\n📊 Training summary:")
        print(f"  Total train time: {train_time_1 + train_time_2:.4f}s")
        print(f"  Forward times: before={fwd_time:.4f}s, after1={fwd_time_after1:.4f}s, after2={fwd_time_after2:.4f}s")

        # stress test: a few big batches through forwardBatch (CPU only)
        print("\n🚀 cpu stress (big batches)")
        big_in  = (c_f * (BATCH_FOR_STRESS * IN_W * IN_H))()
        big_out = (c_f * (BATCH_FOR_STRESS * OUT_SIZE))()
        lib.Paracast_GenerateTestData(big_in, BATCH_FOR_STRESS, 1337)

        print("cpu timing (multiple big batches)…")
        total_ns = 0
        for r in range(STRESS_ROUNDS):
            start_time = time.perf_counter()
            ns = lib.Paracast_ForwardBatch(net, big_in, BATCH_FOR_STRESS, big_out)
            batch_time = time.perf_counter() - start_time
            tps = throughput(BATCH_FOR_STRESS, ns)
            print(f"round {r+1}: {tps:.1f} samples/s  (batch time: {batch_time:.3f}s, measured ns: {ns/1e9:.3f}s)")
            total_ns += ns

        avg_tps = throughput(BATCH_FOR_STRESS * STRESS_ROUNDS, total_ns)
        print(f"\nOverall avg: {avg_tps:.1f} samples/s over {STRESS_ROUNDS} rounds")

        # show a few outputs from the big batch (prove end-to-end)
        print("\n🖨️ sample outputs from big batch:")
        sample0 = [float(big_out[i]) for i in range(OUT_SIZE)]
        print("batch[0] (first 10):", fmt(sample0), f" | argmax = {max(range(OUT_SIZE), key=lambda j: sample0[j])}")

        # Show full outputs for comparison if small
        if OUT_SIZE <= 10:
            print("\nFull output vectors comparison:")
            print("Before train:", fmt(out_before, OUT_SIZE))
            print("After first train:", fmt(out_after_1, OUT_SIZE))
            print("After second train:", fmt(out_after_2, OUT_SIZE))

    finally:
        lib.Paracast_Free(net)

if __name__ == "__main__":
    main()