import json

import os


def format_metrics_for_single_exp(metrics, class_dict):
    table_begin = "\\begin{tabular}{ |c c c c| }\n"
    table_end = "\\end{tabular}\n"

    latex_string = table_begin
    latex_string += "\t\\hline\n"
    latex_string += "\tClass & Precision (\\%) & Recall (\\%) & F1 Score \\\\\n"

    latex_string += "\t\\hline\n"
    for precision, recall, f_score, cl in zip(metrics["precision"], metrics["recall"], metrics["f_score"],
                                             sorted(metrics["classes"])):
        latex_string += f"\t{class_dict[cl]} & ${precision * 100:.02f}$ & ${recall * 100:.02f}$ & ${f_score:.04f}$ \\\\\n"

    latex_string += "\t\\hline\n"
    latex_string += f"\tAverage & ${metrics['w_precision'] * 100:.02f}$ & ${metrics['w_recall'] * 100:.02f}$ & ${metrics['w_f_score']:.04f}$ \\\\\n"

    latex_string += "\t\\hline\n"
    latex_string += table_end

    return latex_string


def format_metrics_for_exp_group(exp_group, relevant_metrics):
    table_begin = "\\begin{tabular}{ |c c c c c| }\n"
    table_end = "\\end{tabular}\n"

    latex_string = table_begin
    latex_string += "\t\\hline\n"

    latex_string += "\tExperiment & Accuracy (\\%) & Precision (\\%) & Recall (\\%) & F1 Score \\\\\n"
    latex_string += "\t\\hline\n"

    for exp, metrics in zip(exp_group, relevant_metrics):
        acc = metrics["accuracy"]
        precision = metrics["w_precision"]
        recall = metrics["w_recall"]
        f_score = metrics["w_f_score"]

        if "_ce" in exp:
            exp_label = "Cross-entropy Loss Classifier"
        elif "_llr" in exp:
            exp_label = "Triplet Loss (Log-likelihood Ratio)"
        elif "full_ds_nums_dates_only" == exp:
            exp_label = "Evaluated on Numbers and Dates"
        elif "full_ds_nums_dates_only_plus_text" == exp:
            exp_label = "Evaluated on Numbers, Dates and Words"
        else:
            exp_label = "Triplet Loss (k-means + kNN)"

        latex_string += f"\t{exp_label} & ${acc * 100:.02f}$ & ${precision * 100:.02f}$ & ${recall * 100:.02f}$ & ${f_score:.04f}$ \\\\\n"

    latex_string += "\t\\hline\n"
    latex_string += table_end

    return latex_string


def main():
    excluded_dirs = ["not", "re", "single_runs"]
    root_dir = "final_runs"

    run_dirs = [path for path in os.listdir(root_dir) if path not in excluded_dirs]
    # run_dirs = ['wpi_on_full_ds_ce', 'full_ds_llr', 'full_ds_nums_dates_only', 'gw_ce', 'full_ds_nums_dates_words_only',
    #  'full_ds_nums_dates_words_only_ce', 'gw_llr', 'gw_baseline', 'full_ds_nums_dates_only_plus_text',
    #  'wpi_on_full_ds_baseline_llr', 'wpi_on_full_ds_baseline', 'full_ds_nums_dates_only_ce', 'full_ds_baseline',
    #  'full_ds_ce']

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
    # TODO: sort run dirs like this?
    plot_dirs = ['full_ds_baseline', 'full_ds_llr', 'full_ds_ce', 'gw_baseline', 'gw_llr', 'gw_ce']
    for run_dir in run_dirs:
        with open(os.path.join(root_dir, run_dir, "metrics.log")) as metric_file:
            metrics = json.load(metric_file)
        all_metrics[run_dir] = metrics

        if run_dir in plot_dirs:
            latex_string = format_metrics_for_single_exp(metrics, class_dict)
            single_exp_tables += run_dir.replace("_", " ") + "\\\\\n" + latex_string + "\n"

    print(single_exp_tables)

    exp_groups = [
        ["gw_baseline", "gw_llr", "gw_ce"],
        ["full_ds_nums_dates_words_only", "full_ds_nums_dates_words_only_ce"],
        ["full_ds_baseline", "full_ds_llr", "full_ds_ce"],
        ["wpi_on_full_ds_baseline", "wpi_on_full_ds_baseline_llr", "wpi_on_full_ds_ce"],
        ["full_ds_nums_dates_only", "full_ds_nums_dates_only_plus_text"]
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
