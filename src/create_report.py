import llm
import sqlite3
import json
import metrics
import numpy as np


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


def create_report(project, model, system_message_file, parameters, k, metric):
    cache = llm.get_chat_completion_cache()

    prompt = []

    system_message = ""
    with open("..//data//" + project + "//system//" + system_message_file + ".txt", "r", encoding="utf-8") as f:
        system_message = f.read()
    for parameter in parameters:
        system_message = system_message.replace("{" + parameter + "}", parameters[parameter])
    prompt.append({
        "role": "system",
        "content": system_message
    })

    conn = sqlite3.connect("..//data//" + project + "//dataset.sqlite3")
    cursor = conn.cursor()
    train = get_dataset(cursor, "train", parameters)
    test = get_dataset(cursor, "test", parameters)
    conn.commit()
    conn.close()

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

    report_name = system_message_file
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
        "metric": metric
    }

    for example in test:
        prompt.append(
            {
                "role": "user",
                "content": example["input"]
            }
        )
        llm_result = llm.call_chatgpt_on_messages(cache, prompt, model=model, temperature=0.0, max_tokens=2000,
                                                  streaming=True)
        example["llm_output"] = llm_result
        example["metric"] = metrics.calculate_metric(example["output"], example["llm_output"], metric)
        prompt.pop()

    test.sort(key=lambda x: x["metric"], reverse=True)
    metric_values = [example["metric"] for example in test]

    average = np.average(metric_values)
    std = np.std(metric_values)
    median = np.median(metric_values)
    percentiles_ = np.percentile(metric_values, [10, 20, 30, 40, 50, 60, 70, 80, 90])
    percentiles = {
        "10": percentiles_[0],
        "20": percentiles_[1],
        "30": percentiles_[2],
        "40": percentiles_[3],
        "50": percentiles_[4],
        "60": percentiles_[5],
        "70": percentiles_[6],
        "80": percentiles_[7],
        "90": percentiles_[8]
    }
    report["average"] = average
    report["median"] = median
    report["std"] = std
    report["percentiles"] = percentiles
    report["n_tests"] = len(test)

    report["prompt"] = prompt
    report["tests"] = test

    json.dump(report, open("..//data//" + project + "//reports//" + report_name + ".json", "w", encoding="utf-8"),
              indent=4,
              ensure_ascii=False)
    print(system_message)


if __name__ == "__main__":
    create_report("grammar_correction", "gpt-3.5-turbo", "improve", {"language": "English"}, 4000, "bleu")
    create_report("grammar_correction", "gpt-3.5-turbo", "improve", {"language": "Spanish"}, 4000, "bleu")
    create_report("grammar_correction", "gpt-3.5-turbo", "improve", {"language": "German"}, 4000, "bleu")
    create_report("grammar_correction", "gpt-3.5-turbo", "improve", {"language": "French"}, 4000, "bleu")
    create_report("grammar_correction", "gpt-3.5-turbo", "improve", {"language": "Russian"}, 2000, "bleu")
    create_report("grammar_correction", "gpt-3.5-turbo", "improve", {"language": "Dutch"}, 4000, "bleu")
    create_report("grammar_correction", "gpt-3.5-turbo", "improve", {"language": "English"}, 2000, "bleu")
    create_report("grammar_correction", "gpt-3.5-turbo", "improve", {"language": "English"}, 1000, "bleu")
    create_report("grammar_correction", "gpt-4", "improve", {"language": "English"}, 1000, "bleu")
    create_report("grammar_correction", "gpt-4", "improve", {"language": "English"}, 0, "bleu")
    create_report("grammar_correction", "gpt-3.5-turbo", "improve", {"language": "English"}, 0, "bleu")
    create_report("grammar_correction", "gpt-4", "improve", {"language": "English"}, 500, "bleu")
    create_report("grammar_correction", "gpt-3.5-turbo", "improve", {"language": "English"}, 500, "bleu")