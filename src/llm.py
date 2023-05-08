import openai
import os
import time
import sqlite3
import json


def get_chat_completion_cache():
    res = {}
    conn = sqlite3.connect("..//data//llm_cache.sqlite3")
    cursor = conn.cursor()
    selection = cursor.execute("SELECT model, prompt, completion output FROM chat_completion")
    selection = selection.fetchall()
    for row in selection:
        res[row[0] + "\n" + row[1]] = row[2]
    conn.commit()
    conn.close()
    return res


def _call_chatgpt_on_messages(messages, model="gpt-3.5-turbo", temperature=0.0, max_tokens=100, streaming=False):
    """
    Calls the ChatGPT API with the given messages, model, temperature, max tokens, and streaming option.
    Prints the role and content of the messages and returns the response from the API.
    """
    print("Calling chatgpt with model " + model + " and temperature " + str(temperature) + "...")
    for message in messages:
        print(message["role"] + ": " + message["content"])
    print("--------")

    completion = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=streaming,
        request_timeout=15
    )

    if streaming:
        res = ""
        for stuff in completion:
            if "content" in stuff["choices"][0]["delta"]:
                res += stuff["choices"][0]["delta"]["content"]
                print("Streamed", stuff["choices"][0]["delta"]["content"])
        print(res)
        return res

    res = completion["choices"][0]["message"]["content"]

    return res


def call_chatgpt_on_messages(cache, messages, model="gpt-3.5-turbo", temperature=0.0, max_tokens=1000, streaming=True):
    if cache is not None:
        key = model + "\n" + json.dumps(messages, indent=4, ensure_ascii=False)
        if key in cache:
            return cache[key]

    openai.api_key = os.environ.get("OPENAI_API_KEY")
    while True:
        try:
            res = _call_chatgpt_on_messages(messages, model=model, temperature=temperature, max_tokens=max_tokens,
                                            streaming=streaming)
            if cache is not None:
                cache[key] = res
                conn = sqlite3.connect("..//data//llm_cache.sqlite3")
                cursor = conn.cursor()
                cursor.execute("INSERT INTO chat_completion (model, prompt, completion) VALUES (?, ?, ?)", (model, json.dumps(messages, indent=4, ensure_ascii=False), res))
                conn.commit()
                conn.close()
            return res
        except Exception as e:
            time.sleep(60)
            print(f"An error occurred: {e}. Retrying...")
