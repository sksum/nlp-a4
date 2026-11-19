import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


QWERTY_NEIGHBORS = {
    "q": "wa", "w": "qeadsz", "e": "wrfsdc", "r": "etgdfx", "t": "ryhgfc", "y": "tujhgv",
    "u": "yikjhb", "i": "uolkjn", "o": "iplkm", "p": "olk",
    "a": "qwezsx", "s": "wedxza", "d": "erfcxs", "f": "rtgvcd", "g": "tyhbvf", "h": "yujnbg",
    "j": "uikmnh", "k": "iolmj", "l": "opk",
    "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb", "b": "vghn", "n": "bhjm", "m": "njk",
}

P_TYPO = 0.25
P_SYN = 0.25


def custom_transform(example):
    tokens = word_tokenize(example["text"])
    transformed = []
    for token in tokens:
        if token.isalpha() and len(token) >= 4 and random.random() < P_TYPO:
            idxs = [i for i in range(1, len(token)) if token[i].lower() in QWERTY_NEIGHBORS]
            if idxs:
                i = random.choice(idxs)
                c = token[i]
                sub = random.choice(QWERTY_NEIGHBORS[c.lower()])
                sub = sub.upper() if c.isupper() else sub
                transformed.append(token[:i] + sub + token[i + 1:])
                continue
        if token.isalpha() and len(token) >= 3 and random.random() < P_SYN:
            lower = token.lower()
            lemmas = [lemma for syn in wordnet.synsets(lower) for lemma in syn.lemma_names()
                      if lemma.isalpha() and lemma.lower() != lower]
            if lemmas:
                replacement = random.choice(lemmas)
                replacement = replacement.capitalize() if token[0].isupper() else replacement
                transformed.append(replacement)
                continue
        transformed.append(token)
    example["text"] = TreebankWordDetokenizer().detokenize(transformed)
    return example
