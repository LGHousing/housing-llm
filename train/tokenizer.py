"""Word-level tokenizer per TASK.md contract.

lowercase; punctuation . , ? ! ' are separate tokens; ; and : map to ",";
hyphens/slashes split words; every other character is dropped.
Specials: <pad>=0 <unk>=1 <bos>=2 <user>=3 <bot>=4 <eos>=5.
"""
import re

SPECIALS = ["<pad>", "<unk>", "<bos>", "<user>", "<bot>", "<eos>"]
PAD, UNK, BOS, USER, BOT, EOS = range(6)

_sub_comma = re.compile(r"[;:]")
_sub_space = re.compile(r"[-/—–]")
_tok = re.compile(r"[a-z0-9]+|[.,?!']")


def words(text):
    t = text.lower().replace("’", "'").replace("‘", "'")
    t = _sub_comma.sub(",", t)
    t = _sub_space.sub(" ", t)
    return _tok.findall(t)


def encode_words(ws, w2i):
    return [w2i.get(w, UNK) for w in ws]


def prompt_ids(question, w2i):
    return [BOS, USER] + encode_words(words(question), w2i) + [BOT]
