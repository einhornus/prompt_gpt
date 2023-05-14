import openai
import os
import time
import sqlite3
import json


def get_chat_completion_cache():
    """
    Fetches chat completion data from a SQLite3 database and returns it in dictionary format. The function reads from
    the 'chat_completion' table in the database and maps each combination of model and prompt to its respective
    completion.

    Returns:
    dict: A dictionary containing cached chat completions. The keys are strings formed by concatenating the model
        name and the prompt with a newline character ("\n") in between. The values are the corresponding completions
        from the database.

    Note:
    This function assumes that there is a SQLite3 database at the path "..//data//llm_cache.sqlite3", and that this
    database contains a table named 'chat_completion' with columns 'model', 'prompt', and 'completion'.
    """
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


def call_chatgpt_on_messages(messages, model="gpt-3.5-turbo", temperature=0.0, max_tokens=100, streaming=False):
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


def call_llm(cache, messages, model="gpt-3.5-turbo", temperature=0.0, max_tokens=1000, streaming=True):
    """
    Fetches responses from the ChatGPT API for a given set of messages, using caching and automatic retries in case of
    errors. If the responses for the same set of messages and model are found in the cache, those are returned instead
    of making another API call. If not, the function makes an API call to fetch the responses and then stores them in
    the cache for future reference.

    Parameters:
    cache (dict): A dictionary used to store the responses. Key is formed by model name and messages. If None, no
                      caching is done.
    messages (list of dict): A list of message-role-content dictionaries that represent a conversation.
    model (str, optional): The ID of the model to use. Defaults to "gpt-3.5-turbo".
    temperature (float, optional): Controls the randomness of the model's output. A higher value makes the output more
                                        random, while a lower value makes it more deterministic. Defaults to 0.0.
    max_tokens (int, optional): The maximum length of the model's output. Defaults to 1000.
    streaming (bool, optional): Whether to stream the output from the model or not. Defaults to True.

    Returns:
    str: The model's output for the given set of messages.

    Raises:
    Exception: Any exception that might occur during the API call. The function will automatically retry after 60 seconds.

    Note:
    This function requires the "OPENAI_API_KEY" environment variable to be set.
    The function also writes to a SQLite3 database located at "..//data//llm_cache.sqlite3".
    """

    if cache is not None:
        key = model + "\n" + json.dumps(messages, indent=4, ensure_ascii=False)
        if key in cache:
            return cache[key]

    openai.api_key = os.environ.get("OPENAI_API_KEY")
    while True:
        try:
            res = call_chatgpt_on_messages(messages, model=model, temperature=temperature, max_tokens=max_tokens,
                                           streaming=streaming)
            if cache is not None:
                cache[key] = res
                conn = sqlite3.connect("..//data//llm_cache.sqlite3")
                cursor = conn.cursor()
                cursor.execute("INSERT INTO chat_completion (model, prompt, completion) VALUES (?, ?, ?)",
                               (model, json.dumps(messages, indent=4, ensure_ascii=False), res))
                conn.commit()
                conn.close()
            return res
        except Exception as e:
            time.sleep(60)
            print(f"An error occurred: {e}. Retrying...")
