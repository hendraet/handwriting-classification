import json

import os


def format_metrics_for_single_exp(metrics, class_dict, run_dir):
    table_begin = "\\begin{table}[h]\n"
    table_begin += "\\centering\n"
    table_begin += "\\begin{tabular}{@{}cccc@{}}\n"
    tabular_end = "\\end{tabular}\n"
    table_end = "\\end{table}\n"

    latex_string = table_begin
    latex_string += "\\toprule\n"
    latex_string += "Class & Precision (\\%) & Recall (\\%) & F1 Score \\\\\n"

    latex_string += "\\midrule\n"
    for precision, recall, f_score, cl in zip(metrics["precision"], metrics["recall"], metrics["f_score"],
                                             sorted(metrics["classes"])):
        latex_string += f"{class_dict[cl]} & {precision * 100:.02f} & {recall * 100:.02f} & {f_score:.04f} \\\\\n"

    latex_string += "\\hline\n"
    latex_string += f"Average & {metrics['w_precision'] * 100:.02f} & {metrics['w_recall'] * 100:.02f} & " \
                    f"{metrics['w_f_score']:.04f} \\\\\n"

    latex_string += "\\bottomrule\n"
    latex_string += tabular_end
    latex_string += "\\caption{}\n"
    latex_string += f"\\label{{table:{run_dir}}}\n"
    latex_string += table_end

    return latex_string


def format_metrics_for_exp_group(exp_group, relevant_metrics):
    table_begin = "\\begin{table}[h]\n"
    table_begin += "\\centering\n"
    table_begin += "\\begin{tabular}{@{}ccccc@{}}\n"
    tabular_end = "\\end{tabular}\n"
    table_end = "\\end{table}\n"

    latex_string = table_begin
    latex_string += "\\toprule\n"

    latex_string += "Experiment & Accuracy (\\%) & Precision (\\%) & Recall (\\%) & F1 Score \\\\\n"
    latex_string += "\\midrule\n"

    for exp, metrics in zip(exp_group, relevant_metrics):
        acc = metrics["accuracy"]
        precision = metrics["w_precision"]
        recall = metrics["w_recall"]
        f_score = metrics["w_f_score"]

        if "_ce" in exp:
            exp_label = "Softmax"
        elif "full_ds_nums_dates_only_llr" == exp:
            exp_label = "LLR two classes"
        elif "full_ds_nums_dates_only_plus_text_llr" == exp:
            exp_label = "LLR three classes"
        elif "_llr" in exp:
            exp_label = "LLR"
        elif "full_ds_nums_dates_only" == exp:
            exp_label = "Naive two classes"
        elif "full_ds_nums_dates_only_plus_text" == exp:
            exp_label = "Naive three classes"
        else:
            exp_label = "Naive"

        latex_string += f"{exp_label} & {acc * 100:.02f} & {precision * 100:.02f} & {recall * 100:.02f} &" \
                        f"{f_score:.04f} \\\\\n"

    latex_string += "\\bottomrule\n"
    latex_string += tabular_end
    latex_string += "\\caption{}\n"
    latex_string += f"\\label{{table:}}\n"
    latex_string += table_end

    return latex_string


def main():
    # excluded_dirs = ["not", "re", "single_runs"]
    excluded_dirs = ["not"]
    root_dir = "../../final_runs"

    # run_dirs = [path for path in os.listdir(root_dir) if path not in excluded_dirs]
    run_dirs = [
        # gan
        'gw_baseline',
        'gw_llr',
        'gw_ce',
        # comp
        'full_ds_nums_dates_words_only',
        'full_ds_nums_dates_words_only_llr',
        'full_ds_nums_dates_words_only_ce',
        # main
        'full_ds_baseline',
        'full_ds_llr',
        'full_ds_ce',
        # unseen data
        'wpi_on_full_ds_baseline',
        'wpi_on_full_ds_llr',
        'wpi_on_full_ds_ce',
        # additional class
        'full_ds_nums_dates_only',
        'full_ds_nums_dates_only_llr',
        'full_ds_nums_dates_only_plus_text',
        'full_ds_nums_dates_only_plus_text_llr'
    ]

    class_dict = {
        "alpha_num": "Alphanumeric",
        "alphanum": "Alphanumeric",
        "date": "Date",
        "num":  "Number",
        "plz":  "Zip Code",
        "text": "Word"
    }

    single_exp_tables = ""

    all_metrics = {}
    plot_dirs = run_dirs
    for run_dir in run_dirs:
        with open(os.path.join(root_dir, run_dir, "metrics.log")) as metric_file:
            metrics = json.load(metric_file)
        all_metrics[run_dir] = metrics

        if run_dir in plot_dirs:
            latex_string = format_metrics_for_single_exp(metrics, class_dict, run_dir)
            single_exp_tables += run_dir.replace("_", " ") + "\\\\\n" + latex_string + "\n"

    print(single_exp_tables)

    exp_groups = [
        ["gw_baseline", "gw_llr", "gw_ce"],
        ["full_ds_nums_dates_words_only", "full_ds_nums_dates_words_only_llr", "full_ds_nums_dates_words_only_ce"],
        ["full_ds_baseline", "full_ds_llr", "full_ds_ce"],
        ["wpi_on_full_ds_baseline", "wpi_on_full_ds_llr", "wpi_on_full_ds_ce"],
        ["full_ds_nums_dates_only", "full_ds_nums_dates_only_llr", "full_ds_nums_dates_only_plus_text", "full_ds_nums_dates_only_plus_text_llr"]
    ]

    exp_group_tables = ""

    for exp_group in exp_groups:
        relevant_metrics = [all_metrics[exp] for exp in exp_group]

        latex_string = format_metrics_for_exp_group(exp_group, relevant_metrics)

        exp_group_tables += ", ".join(exp_group).replace("_", " ") + "\\\\\n" + latex_string + "\n"

    print(exp_group_tables)

    with open("/home/hendraet/stud_sync/Studium/master/ma/thesis/tables.tex", "w") as table_file:
        table_file.write(single_exp_tables)
        table_file.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n")
        table_file.write(exp_group_tables)


if __name__ == '__main__':
    main()
