using System;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;

static class Native
{
    // Adjust the Library name if you move it. On Linux, the loader will search:
    // - process working dir
    // - LD_LIBRARY_PATH
    // - system lib paths
    const string Lib = "libparacast.so";

    // C ABI (cdecl) imports
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    public static extern long Paracast_CreateLargeNetwork();

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    public static extern void Paracast_DisableGPU(long net);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    public static extern void Paracast_GenerateTestData(float[] buf, int samples, int seed);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    public static extern int Paracast_Forward(long net, float[] input, int inW, int inH);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    public static extern int Paracast_GetOutput(long net, float[] output, int outSize);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    public static extern long Paracast_ForwardBatch(long net, float[] input, int nSamples, float[] output);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    public static extern int Paracast_Train(
        long net,
        float[] inputs, int inW, int inH,
        float[] targets, int outW, int outH,
        int batch,
        int epochs,
        float lr,
        int useGPU,
        float clipLower, float clipUpper
    );

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr Paracast_GetLastError();

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    public static extern void Paracast_Free(long net);

    public static string LastError()
    {
        var ptr = Paracast_GetLastError();
        return ptr == IntPtr.Zero ? "?" : Marshal.PtrToStringAnsi(ptr) ?? "?";
    }

    public static void Check(int ok, string where)
    {
        if (ok == 0) throw new Exception($"{where} failed: {LastError()}");
    }
}

class Program
{
    static string Fmt(float[] v, int n = 10)
        => "[" + string.Join(", ", v.Take(n).Select(x => x.ToString("0.000"))) + (v.Length > n ? ", ..." : "") + "]";

    static double Throughput(long samples, long ns) => ns > 0 ? samples / (ns / 1e9) : double.NaN;

    static void Main()
    {
        // Match python constants
        const int IN_W = 28, IN_H = 28;
        const int OUT_W = 10, OUT_H = 1;
        const int OUT_SIZE = OUT_W * OUT_H;

        const double SEED_SAMPLE_NOISE = 0.02;
        const int NUDGE_CLASS_1 = 9;
        const int NUDGE_CLASS_2 = 0;
        const int EPOCHS_TRAIN_1 = 10;
        const int EPOCHS_TRAIN_2 = 10;
        const float LR_TRAIN = 0.1f;

        const int BATCH_FOR_STRESS = 4096;
        const int STRESS_ROUNDS = 3;

        Console.WriteLine("🔧 create model");
        long net = Native.Paracast_CreateLargeNetwork();
        if (net == 0) throw new Exception("Paracast_CreateLargeNetwork failed: " + Native.LastError());

        try
        {
            // Deterministic single input
            var rnd = new Random(42);
            var one_in = new float[IN_W * IN_H];
            for (int i = 0; i < one_in.Length; i++) one_in[i] = (float)rnd.NextDouble();

            // Forward BEFORE any training
            Console.WriteLine("\n▶️  forward (cpu) — before any training");
            var sw = Stopwatch.StartNew();
            Native.Check(Native.Paracast_Forward(net, one_in, IN_W, IN_H), "Paracast_Forward(before)");
            sw.Stop();
            var out_before = new float[OUT_SIZE];
            Native.Check(Native.Paracast_GetOutput(net, out_before, OUT_SIZE), "Paracast_GetOutput(before)");
            int argmax_before = Array.IndexOf(out_before, out_before.Max());
            Console.WriteLine($"forward time: {sw.Elapsed.TotalSeconds:0.0000}s");
            Console.WriteLine($"output (first 10): {Fmt(out_before)}  | argmax = {argmax_before}");

            // Training batch
            int B = 128;
            var batch_x = new float[B * IN_W * IN_H];
            var batch_y = new float[B * OUT_SIZE];

            for (int b = 0; b < B; b++)
            {
                for (int i = 0; i < IN_W * IN_H; i++)
                {
                    batch_x[b * IN_W * IN_H + i] =
                        one_in[i] + (float)(SEED_SAMPLE_NOISE * (rnd.NextDouble() - 0.5));
                }
            }
            // nudge to class 9
            for (int b = 0; b < B; b++)
            {
                int baseIdx = b * OUT_SIZE;
                for (int j = 0; j < OUT_SIZE; j++)
                    batch_y[baseIdx + j] = (j == NUDGE_CLASS_1) ? 1f : 0f;
            }

            // First train (CPU)
            Console.WriteLine($"\n🎓 first train (cpu) — nudging to class {NUDGE_CLASS_1}, {EPOCHS_TRAIN_1} epochs, LR={LR_TRAIN}");
            sw.Restart();
            int ok1 = Native.Paracast_Train(
                net,
                batch_x, IN_W, IN_H,
                batch_y, OUT_W, OUT_H,
                B, EPOCHS_TRAIN_1,
                LR_TRAIN,
                0,           // CPU
                -1e9f, 1e9f
            );
            sw.Stop();
            Native.Check(ok1, "Paracast_Train(first)");
            Console.WriteLine($"train time: {sw.Elapsed.TotalSeconds:0.0000}s");

            // Forward AFTER first
            Console.WriteLine("▶️  forward (cpu) — after first train");
            sw.Restart();
            Native.Check(Native.Paracast_Forward(net, one_in, IN_W, IN_H), "Paracast_Forward(after first)");
            sw.Stop();
            var out_after1 = new float[OUT_SIZE];
            Native.Check(Native.Paracast_GetOutput(net, out_after1, OUT_SIZE), "Paracast_GetOutput(after first)");
            int argmax_after1 = Array.IndexOf(out_after1, out_after1.Max());
            Console.WriteLine($"forward time: {sw.Elapsed.TotalSeconds:0.0000}s");
            Console.WriteLine($"output (first 10): {Fmt(out_after1)}  | argmax = {argmax_after1}");
            Console.WriteLine($"Change in argmax: {argmax_before} -> {argmax_after1}");

            // Second training: nudge to class 0
            for (int b = 0; b < B; b++)
            {
                int baseIdx = b * OUT_SIZE;
                for (int j = 0; j < OUT_SIZE; j++)
                    batch_y[baseIdx + j] = (j == NUDGE_CLASS_2) ? 1f : 0f;
            }

            Console.WriteLine($"\n🎓 second train (cpu) — now nudging to class {NUDGE_CLASS_2}, {EPOCHS_TRAIN_2} epochs, LR={LR_TRAIN}");
            sw.Restart();
            int ok2 = Native.Paracast_Train(
                net,
                batch_x, IN_W, IN_H,
                batch_y, OUT_W, OUT_H,
                B, EPOCHS_TRAIN_2,
                LR_TRAIN,
                0,
                -1e9f, 1e9f
            );
            sw.Stop();
            Native.Check(ok2, "Paracast_Train(second)");
            Console.WriteLine($"train time: {sw.Elapsed.TotalSeconds:0.0000}s");

            // Forward AFTER second
            Console.WriteLine("▶️  forward (cpu) — after second train");
            sw.Restart();
            Native.Check(Native.Paracast_Forward(net, one_in, IN_W, IN_H), "Paracast_Forward(after second)");
            sw.Stop();
            var out_after2 = new float[OUT_SIZE];
            Native.Check(Native.Paracast_GetOutput(net, out_after2, OUT_SIZE), "Paracast_GetOutput(after second)");
            int argmax_after2 = Array.IndexOf(out_after2, out_after2.Max());
            Console.WriteLine($"forward time: {sw.Elapsed.TotalSeconds:0.0000}s");
            Console.WriteLine($"output (first 10): {Fmt(out_after2)}  | argmax = {argmax_after2}");
            Console.WriteLine($"Change in argmax: {argmax_after1} -> {argmax_after2}");

            // Stress: ForwardBatch
            Console.WriteLine("\n🚀 cpu stress (big batches)");
            var big_in  = new float[BATCH_FOR_STRESS * IN_W * IN_H];
            var big_out = new float[BATCH_FOR_STRESS * OUT_SIZE];
            Native.Paracast_GenerateTestData(big_in, BATCH_FOR_STRESS, 1337);

            long totalNs = 0;
            for (int r = 0; r < STRESS_ROUNDS; r++)
            {
                var swWall = Stopwatch.StartNew();
                long ns = Native.Paracast_ForwardBatch(net, big_in, BATCH_FOR_STRESS, big_out);
                swWall.Stop();
                double tps = Throughput(BATCH_FOR_STRESS, ns);
                Console.WriteLine($"round {r+1}: {tps:0.0} samples/s  (batch time: {swWall.Elapsed.TotalSeconds:0.000}s, measured ns: {ns/1e9:0.000}s)");
                totalNs += ns;
            }
            double avg = Throughput(BATCH_FOR_STRESS * STRESS_ROUNDS, totalNs);
            Console.WriteLine($"\nOverall avg: {avg:0.0} samples/s over {STRESS_ROUNDS} rounds");

            // Sample outputs from big batch
            var sample0 = new float[OUT_SIZE];
            Array.Copy(big_out, 0, sample0, 0, OUT_SIZE);
            int arg0 = Array.IndexOf(sample0, sample0.Max());
            Console.WriteLine("\n🖨️ sample outputs from big batch:");
            Console.WriteLine($"batch[0] (first 10): {Fmt(sample0)}  | argmax = {arg0}");

            if (OUT_SIZE <= 10)
            {
                Console.WriteLine("\nFull output vectors comparison:");
                Console.WriteLine($"Before train:       {Fmt(out_before, OUT_SIZE)}");
                Console.WriteLine($"After first train:  {Fmt(out_after1, OUT_SIZE)}");
                Console.WriteLine($"After second train: {Fmt(out_after2, OUT_SIZE)}");
            }
        }
        finally
        {
            Native.Paracast_Free(net);
        }
    }
}

