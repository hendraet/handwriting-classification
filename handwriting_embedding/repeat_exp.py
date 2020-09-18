import glob
import subprocess
import sys

import os


def main():
    # Orig
    # exp_dirs = ["full_ds_baseline_ep20", "full_ds_nums_dates_only_ep20", "full_ds_nums_dates_only_plus_text_eval",
    #             "full_ds_nums_dates_words_only_ep20", "gw_baseline_ep20", "wpi_on_full_ds_baseline_eval"]
    exp_dirs = ["full_ds_baseline_ep20", "full_ds_nums_dates_only_ep20", "full_ds_nums_dates_only_plus_text_eval",
                "full_ds_nums_dates_words_only_ep20", "gw_baseline_ep20", "wpi_on_full_ds_baseline_eval"]
    root_dir = "final_runs/re"
    exp_dirs = [os.path.join(root_dir, p) for p in exp_dirs]
    assert False, "script is likely broken, fix first"

    use_llr = False
    if use_llr:
        print("Evaluating using log-likelihood ratios")
        assert False, "Manual renaming possible at the end"

    log_dir = "runs"

    environ = os.environ.copy()
    print(f"CUDA DEVICE: {environ['CUDA_VISIBLE_DEVICES']}")

    for path in exp_dirs:
        print(f"Processing: {path}")
        conf = os.path.join(path, "own.conf")
        with open(os.path.join(path, "args.log")) as af:
            args = af.readline().split()[1:]

        if "plus" in path:
            pretrained_model = glob.glob(os.path.join(exp_dirs[0].split("_plus")[0], "*_best.npz"))[0]
        elif "wpi" in path:
            dir = os.path.join(root_dir, [f for f in os.listdir(root_dir) if exp_dirs[0].split("on_")[1].split("_eval")[0] in f][0])
            pretrained_model = glob.glob(os.path.join(dir, "*_best.npz"))[0]
        else:
            pretrained_model = glob.glob(os.path.join(path, "*_best.npz"))[0]
        cmd = [sys.executable, "train_own.py", conf, *args, "-eo", pretrained_model]

        if use_llr:
            cmd.append("-llr")

        log_dir_filename = os.path.join(log_dir, f"{args[0]}.log")
        with open(log_dir_filename, "w") as log_f:
            subprocess.run(cmd, stdout=log_f, env=environ)


if __name__ == '__main__':
    main()
