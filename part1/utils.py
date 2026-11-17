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


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    import re

    rng = random.Random(42)
    parts = re.findall(
        r"<[^>]*>|[A-Za-z]+(?:['’][A-Za-z]+)*|\d+|\s+|.",
        example["text"], flags=re.DOTALL,
    )

    protected = {
        "not", "no", "never", "without", "neither", "nor",
        "very", "extremely", "barely", "hardly", "really",
    }

    synonyms = {
        "movie": "film", "movies": "films",
        "perhaps": "maybe", "maybe": "perhaps",
    }
    keyboard = {
        "q": "wa", "w": "qeas", "e": "wrsd", "r": "etdf",
        "t": "ryfg", "y": "tugh", "u": "yihj", "i": "uojk",
        "o": "ipkl", "p": "ol", "a": "qwsz", "s": "weadzx",
        "d": "ersfxc", "f": "rtdgcv", "g": "tyfhvb", "h": "yugjbn",
        "j": "uihknm", "k": "iojlm", "l": "opk", "z": "asx",
        "x": "zsdc", "c": "xdfv", "v": "cfgb", "b": "vghn",
        "n": "bhjm", "m": "njk",
    }

    def make_typo(word):

        options = {"swap": [], "keyboard": []}
        for j in range(1, len(word) - 2):
            if word[j] != word[j + 1]:
                options["swap"].append(
                    word[:j] + word[j + 1] + word[j] + word[j + 2:]
                )
        for j in range(1, len(word) - 1):
            for letter in keyboard[word[j]]:
                options["keyboard"].append(word[:j] + letter + word[j + 1:])
        methods = ["swap", "keyboard"]
        rng.shuffle(methods)
        for method in methods:
            rng.shuffle(options[method])
            for changed in options[method]:
                if changed not in protected and not wordnet.synsets(changed):
                    return changed
        return word

    words = [i for i, part in enumerate(parts)
             if re.fullmatch(r"[A-Za-z]+(?:['’][A-Za-z]+)*", part)]
    candidates = [i for i in words
                  if re.fullmatch(r"[a-z]{5,}", parts[i])
                  and parts[i] not in protected]
    rng.shuffle(candidates)

    budget = int(len(words) * 0.18)
    ranks = {index: rank for rank, index in enumerate(words)}
    selected = set()
    for i in candidates:
        if len(selected) >= budget:
            break
        rank = ranks[i]
        if rank - 1 in selected or rank + 1 in selected:
            continue
        original = parts[i]
        method = rng.choices(["synonym", "typo", "both"],
                             weights=[0.2, 0.6, 0.2], k=1)[0]
        changed = synonyms.get(original, original) if method != "typo" else original
        if method in ("typo", "both") or changed == original:
            changed = make_typo(changed)
        if changed != original:
            parts[i] = changed
            selected.add(rank)
    example["text"] = "".join(parts)

    ##### YOUR CODE ENDS HERE ######

    return example
