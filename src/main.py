import os
import llm
import sqlite3

def fix(text, language):
    prompt = [
        {
            "role": "system",
            "content": "You correct spelling and punctuation mistakes in the given " + language + " text. Please only print the corrected text, nothing else.",
        },
        {
            "role": "user",
            "content": text,
        }
    ]
    result = utils.call_chatgpt_on_messages(prompt, model="gpt-4", temperature=0.0, max_tokens=1000)
    return result

"""
def main(lang):
    files = os.listdir("..//data//correction//train")
    for f in files:
        if f.startswith(lang):
            content = open("..//data//correction//train//" + f, "r", encoding="utf-8").read()
            print(content)
            parts = content.split("\n----------\n")

            fixed_part = fix(parts[2], lang)

            new_content = parts[0] + "\n----------\n" + parts[1] + "\n----------\n" + fixed_part
            with open("..//data//correction//train//" + f, "w", encoding="utf-8") as f:
                f.write(new_content)
"""

def main(lang):
    files = os.listdir("..//data//correction//test")
    for f in files:
        if f.startswith(lang):
            content = open("..//data//correction//test//" + f, "r", encoding="utf-8").read()
            print(content)
            parts = content.split("\n----------\n")

            fixed_part = fix(parts[2], lang)

            new_content = parts[0] + "\n----------\n" + parts[1] + "\n----------\n" + fixed_part
            with open("..//data//correction//test//" + f, "w", encoding="utf-8") as f:
                f.write(new_content)

if __name__ == "__main__":
    #main("Russian")
    #main("Dutch")
    #main("English")
    #main("Spanish")
    #main("French")
    #main("German")

    conn = sqlite3.connect('..//data//correction//dataset.sqlite3')
    cursor = conn.cursor()

    files = os.listdir("..//data//correction//test")
    files.sort(key = lambda x: x.split("_")[0] + (x.split("_")[-2].replace('.txt', '')))

    id = 1
    for f in files:
        content = open("..//data//correction//test//" + f, "r", encoding="utf-8").read()
        print(content)
        parts = content.split("\n----------\n")
        name = f.replace('_.txt', '')
        input = parts[1]
        output = parts[2]
        parameters = parts[0]
        data_to_insert = (id, name, parameters, input, output)
        id += 1
        cursor.execute('INSERT INTO test (id, name, parameters, input, output) VALUES (?, ?, ?, ?, ?)', data_to_insert)

    conn.commit()
    conn.close()



    conn = sqlite3.connect('..//data//correction//dataset.sqlite3')
    cursor = conn.cursor()


    files = os.listdir("..//data//correction//train")
    files.sort(key = lambda x: x.split("_")[0] + (x.split("_")[-2].replace('.txt', '')))

    id = 1
    for f in files:
        content = open("..//data//correction//train//" + f, "r", encoding="utf-8").read()
        print(content)
        parts = content.split("\n----------\n")
        name = f.replace('_.txt', '')
        input = parts[1]
        output = parts[2]
        parameters = parts[0]
        data_to_insert = (id, name, parameters, input, output)
        id += 1
        cursor.execute('INSERT INTO train (id, name, parameters, input, output) VALUES (?, ?, ?, ?, ?)', data_to_insert)

    conn.commit()
    conn.close()



