import llm
import sqlite3
import json
import metrics
import numpy as np
import random


def check_parameters(parameters, parameters_value):
    parameters_set = parameters_value.split("\n")
    for line in parameters_set:
        key, value = line.split("=")
        if key not in parameters:
            return False
        if parameters[key] != value:
            return False
    return True


def get_dataset(cursor, table_name, parameters):
    res = []
    train = cursor.execute("SELECT id, name, parameters, input, output FROM " + table_name)
    train = train.fetchall()
    for row in train:
        _id = row[0]
        name = row[1]
        parameters_value = row[2]
        _input = row[3]
        output = row[4]
        if check_parameters(parameters, parameters_value):
            obj = {
                "id": _id,
                "name": name,
                "input": _input,
                "output": output
            }
            res.append(obj)
    return res


def create_report(project, model, system_message_file, parameters, k, metric, n=1):
    """
    This function generates a report for a given project using provided parameters and evaluates it with a given
    metric. It first reads a system message from a file, substitutes parameters in this message, and then
    retrieves training and testing data from a SQLite database. It uses training data to create a prompt for a
    language model and uses testing data to evaluate the model's performance. Finally, it saves the best report
    to a JSON file.

    Parameters:
    -----------
    project : str
        Name of the project. This is used to determine the directory path of the system message file and the
        SQLite database.

    model : str
        The name of the model to be evaluated.

    system_message_file : str
        The name of the file containing the system message (without the .txt extension). This file should be
        located in the 'system' subdirectory under the project directory.

    parameters : dict
        A dictionary of parameter names and values to be substituted into the system message.

    k : int
        The maximum size (in number of characters) of the chat history examples
        to be used in the prompt for the language model.

    metric : str
        The name of the metric to be used for evaluation.

    n : int, optional
        The number of times to repeat the whole process (default is 1).

    Returns:
    --------
    None. The function saves the generated report in a JSON file in the 'reports' subdirectory under the project
    directory. The name of the report file is determined by the system message file name, the parameters, the
    model name, the value of k, and the metric name.

    Raises:
    -------
    FileNotFoundError: If the system message file or the SQLite database file does not exist.
    sqlite3.OperationalError: If there is a problem with the SQLite database operations.
    """

    cache = llm.get_chat_completion_cache()

    history = []
    reports = []

    for q in range(n):
        system_message = ""
        with open("..//data//" + project + "//system//" + system_message_file + ".txt", "r", encoding="utf-8") as f:
            system_message = f.read()
        for parameter in parameters:
            system_message = system_message.replace("{" + parameter + "}", parameters[parameter])

        prompt = [{
            "role": "system",
            "content": system_message
        }]

        conn = sqlite3.connect("..//data//" + project + "//dataset.sqlite3")
        cursor = conn.cursor()
        train = get_dataset(cursor, "train", parameters)
        test = get_dataset(cursor, "test", parameters)
        conn.commit()
        conn.close()

        if q > 0:
            random.seed(q)
            random.shuffle(train)

        examples_total_size = 0
        for example in train:
            if examples_total_size + len(example["input"]) + len(example["output"]) > k:
                continue
            prompt.append({
                "role": "assistant",
                "content": example["output"]
            })
            prompt.append({
                "role": "user",
                "content": example["input"]
            })
            examples_total_size += len(example["input"]) + len(example["output"])
        prompt.reverse()

        report_name = "system=" + system_message_file
        parameters_keys = list(parameters.keys())
        parameters_keys.sort()
        for parameter in parameters_keys:
            report_name += " " + parameter + "=" + parameters[parameter] + " "
        report_name += " model=" + model + " k=" + str(k) + " metric=" + metric

        report = {
            "system_message_file": system_message_file,
            "model": model,
            "parameters": parameters,
            "k": k,
            "metric": metric,
            "history": history,
        }

        for example in test:
            prompt.append(
                {
                    "role": "user",
                    "content": example["input"]
                }
            )
            llm_result = llm.call_llm(cache, prompt, model=model, temperature=0.0, max_tokens=2000,
                                      streaming=True)
            example["llm_output"] = llm_result
            example["score"] = metrics.calculate_metric(example["output"], example["llm_output"], metric)
            prompt.pop()

        test.sort(key=lambda x: x["score"], reverse=True)
        metric_values = [example["score"] for example in test]

        average = np.average(metric_values)
        std = np.std(metric_values)
        median = np.median(metric_values)
        percentiles_ = np.percentile(metric_values, [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90])
        percentiles = {
            "10": percentiles_[0],
            "20": percentiles_[1],
            "25": percentiles_[2],
            "30": percentiles_[3],
            "40": percentiles_[4],
            "50": percentiles_[5],
            "60": percentiles_[6],
            "70": percentiles_[7],
            "75": percentiles_[8],
            "80": percentiles_[9],
            "90": percentiles_[10]
        }
        report["average"] = average
        report["median"] = median
        report["std"] = std
        report["percentiles"] = percentiles
        report["n_tests"] = len(test)
        report["history"].append(average)
        report["prompt"] = prompt
        report["tests"] = test

        reports.append(report)
        reports.sort(key=lambda x: x["average"], reverse=True)

        json.dump(reports[0],
                  open("..//data//" + project + "//reports//" + report_name + ".json", "w", encoding="utf-8"),
                  indent=4,
                  ensure_ascii=False)


if __name__ == "__main__":
    for i in range(1, 7):
        create_report("grammar_correction", "gpt-3.5-turbo", str(i), {"language": "English"}, 2000, "bleu", 10)
        create_report("grammar_correction", "gpt-3.5-turbo", str(i), {"language": "English"}, 0, "bleu", 10)

    create_report("grammar_correction", "gpt-3.5-turbo", "4", {"language": "English"}, 500, "bleu", 10)
    create_report("grammar_correction", "gpt-3.5-turbo", "4", {"language": "English"}, 1000, "bleu", 10)
    create_report("grammar_correction", "gpt-3.5-turbo", "4", {"language": "English"}, 2000, "bleu", 10)

    create_report("grammar_correction", "gpt-4", "4", {"language": "English"}, 0, "bleu", 10)

    create_report("grammar_correction", "gpt-3.5-turbo", "4", {"language": "Russian"}, 0, "bleu", 10)
    create_report("grammar_correction", "gpt-3.5-turbo", "4", {"language": "Spanish"}, 0, "bleu", 10)
    create_report("grammar_correction", "gpt-3.5-turbo", "4", {"language": "French"}, 0, "bleu", 10)
    create_report("grammar_correction", "gpt-3.5-turbo", "4", {"language": "German"}, 0, "bleu", 10)
    create_report("grammar_correction", "gpt-3.5-turbo", "4", {"language": "Dutch"}, 0, "bleu", 10)
