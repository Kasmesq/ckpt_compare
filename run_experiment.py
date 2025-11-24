import os
import subprocess
import datetime

def main():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = f"exp_{ts}"
    os.makedirs(outdir, exist_ok=True)

    print("=== Running training ===")
    subprocess.run([
        "deepspeed",
        "--num_gpus", "4",
        "datastates_train_bloom3b.py",
        "--deepspeed_config", "ds_config_zero2_datastates.json",
        "--train_file", "input_data.txt",
        "--output_dir", outdir,
        "--epochs", "1"
    ])

    print("=== Generating plots ===")
    subprocess.run(["python", "plot_metrics.py"])

    print("Done. Experiment output:", outdir)


if __name__ == "__main__":
    main()
