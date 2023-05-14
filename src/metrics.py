from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize


def calculate_metric(expected, actual, metric):
    if metric == "bleu":
        return calculate_bleu(expected, actual)


def calculate_bleu(expected, actual):
    tokenized_reference = word_tokenize(expected.lower())
    tokenized_candidate = word_tokenize(actual.lower())
    tokenized_reference = [e for e in tokenized_reference if e not in ('.', ',', '?', '!')]
    tokenized_candidate = [e for e in tokenized_candidate if e not in ('.', ',', '?', '!')]
    reference_list = [tokenized_reference]
    bleu_score = sentence_bleu(reference_list, tokenized_candidate)
    return bleu_score

