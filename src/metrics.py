from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize


def calculate_metric(expected, actual, metric):
    """
    This function calculates the specified metric for comparing two sentences.

    Parameters:
    expected (str): The original sentence.
    actual (str): The generated sentence.
    metric (str): The type of metric to calculate. Currently, only 'bleu' is supported.

    Returns:
    float: The calculated metric. Returns None if the metric type is not supported.
    """
    if metric == "bleu":
        return calculate_bleu(expected, actual)


def calculate_bleu(expected, actual):
    """
    This function calculates the BLEU score for comparing two sentences.

    Parameters:
    expected (str): The original sentence.
    actual (str): The generated sentence.

    Returns:
    float: The BLEU score for the two sentences. A higher score indicates a closer match to the original sentence.

    Note:
    Punctuation marks ('.', ',', '?', '!') are removed before calculating the BLEU score.
    """
    tokenized_reference = word_tokenize(expected.lower())
    tokenized_candidate = word_tokenize(actual.lower())
    tokenized_reference = [e for e in tokenized_reference if e not in ('.', ',', '?', '!')]
    tokenized_candidate = [e for e in tokenized_candidate if e not in ('.', ',', '?', '!')]
    reference_list = [tokenized_reference]
    bleu_score = sentence_bleu(reference_list, tokenized_candidate)
    return bleu_score

