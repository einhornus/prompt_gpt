import matplotlib.pyplot as plt
import os
import json
import numpy as np


def visualize(project, system_message_files, models, parameter_configurations, ks, metric):
    report_files = os.listdir("..//data//" + project + "//reports//")
    report_files.sort(key = lambda x: int(x.split(" ")[-2].replace("k=", "")))

    data = []
    labels = []

    system_messages_set = set()
    models_set = set()
    parameter_configurations_set = set()
    ks_set = set()

    for report_file in report_files:
        report = json.load(open("..//data//" + project + "//reports//" + report_file, "r", encoding="utf-8"))
        if report["metric"] == metric:
            if report["system_message_file"] in system_message_files:
                if report["model"] in models:
                    if report["parameters"] in parameter_configurations:
                        if report["k"] in ks:
                            data_column = []
                            for test in report["tests"]:
                                data_column.append(test["metric"])
                            data.append(data_column)
                            system_messages_set.add(report["system_message_file"])
                            models_set.add(report["model"])
                            string_parameter_list = ""
                            keys = list(report["parameters"].keys())
                            keys.sort()
                            for i in range(len(keys)):
                                string_parameter_list += (keys[i][0] + "=" + report["parameters"][keys[i]])
                                if i < len(keys) - 1:
                                    string_parameter_list += "; "
                            parameter_configurations_set.add(string_parameter_list)
                            ks_set.add(report["k"])
                            labels.append(
                                (report["system_message_file"], report["model"], string_parameter_list, report["k"], report["average"], report["std"], report["median"]))

    common_label = project + "\n"
    if len(system_messages_set) == 1:
        common_label += "system="+list(system_messages_set)[0] + " "
    if len(models_set) == 1:
        common_label += "model="+list(models_set)[0] + " "
    if len(parameter_configurations_set) == 1:
        common_label += "p={"+list(parameter_configurations_set)[0] + "} "
    if len(ks_set) == 1:
        common_label += "k="+str(list(ks_set)[0]) + " "

    string_labels = []
    for label in labels:
        string_label = f"{label[4]:.3f} ± {label[5]:.3f}\n"
        string_label += f"med = {label[6]:.3f}\n"
        if len(system_messages_set) > 1:
            string_label += "system="+label[0] + " "
        if len(models_set) > 1:
            string_label += "model="+label[1] + " "
        if len(parameter_configurations_set) > 1:
            string_label += "p={"+label[2] + "} "
        if len(ks_set) > 1:
            string_label += "k="+str(label[3]) + " "
        string_labels.append(string_label)

    plt.figure(figsize=(10, 6))
    plt.style.use('ggplot')
    bp = plt.boxplot(data, patch_artist=True, notch=True)
    colors = ['lightblue', 'lightgreen', 'pink', 'purple']
    for element in ['whiskers', 'caps', 'fliers', 'medians']:
        plt.setp(bp[element], color='black')
    plt.xticks(np.arange(1, len(labels) + 1), string_labels)
    plt.title(common_label)
    for i in range(len(labels)):
        plt.plot([], label=string_labels[i])
    # plt.legend()
    plt.show()


if __name__ == "__main__":
    """
    visualize("grammar_correction",
              ["improve"],
              ["gpt-3.5-turbo"],
              [
                  {"language": "English"},
                  {"language": "German"},
                  {"language": "French"},
                  {"language": "Spanish"},
                  {"language": "Dutch"}
              ],
              [4000],
              "bleu"
              )
    """

    """
    visualize("grammar_correction",
              ["improve"],
              ["gpt-3.5-turbo"],
              [
                  {"language": "English"},
              ],
              [0, 500, 1000, 2000, 4000],
              "bleu"
              )
    """

    """
    visualize("grammar_correction",
              ["improve"],
              ["gpt-3.5-turbo", "gpt-4"],
              [
                  {"language": "English"},
              ],
              [0],
              "bleu"
              )
    """

    visualize("grammar_correction",
              ["improve"],
              ["gpt-4"],
              [
                  {"language": "English"},
              ],
              [0, 500, 1000],
              "bleu"
              )