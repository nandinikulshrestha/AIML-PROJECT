import os, sys, re
os.system('cls' if os.name == 'nt' else 'clear')

# --- Optional: keep NLTK quiet + ensure tokenizers available
import nltk
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt', quiet=True)
    try:
        nltk.download('punkt_tab', quiet=True)
    except Exception:
        pass

from autocorrect import Speller
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# ---------- 1) Pre-pass: spelling ----------
spell = Speller(lang='en')

def quick_spell_fix(text: str) -> str:
    # token-level spell correction but keep punctuation/numbers/emojis intact
    tokens = re.findall(r"[A-Za-z]+|[^A-Za-z\s]", text)
    out = []
    for t in tokens:
        if re.fullmatch(r"[A-Za-z]+", t):
            fixed = spell(t)
            # preserve simple casing styles
            if t.isupper(): fixed = fixed.upper()
            elif t[0].isupper(): fixed = fixed.capitalize()
            out.append(fixed)
        else:
            out.append(t)
    # naive detokenize: collapse spaces around punctuation
    s = " ".join(out)
    s = re.sub(r"\s+([.,!?;:])", r"\1", s)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    s = re.sub(r"\s+'", "'", s)
    return s

# ---------- 2) Grammar model (Transformer) ----------
# We’ll try a couple of popular grammar models and pick the first that works.
MODEL_CANDIDATES = [
    # widely used T5-based GEC checkpoints
    "vennify/t5-base-grammar-correction",
    "prithivida/grammar_error_correcter_v1",
]

gec_pipe = None
last_error = None
for mid in MODEL_CANDIDATES:
    try:
        tokenizer = AutoTokenizer.from_pretrained(mid)
        model = AutoModelForSeq2SeqLM.from_pretrained(mid)
        gec_pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
        active_model = mid
        break
    except Exception as e:
        last_error = e
        continue

if gec_pipe is None:
    print("Could not load a grammar model. Error:", last_error)
    print("Try: pip install --upgrade transformers torch sentencepiece accelerate")
    sys.exit(1)

def grammar_correct(text: str, max_len=256) -> str:
    # Some models expect a prompt like "gec: <text>"
    inp = text
    if "grammar_error_correcter" in active_model:
        inp = f"gec: {text}"
    result = gec_pipe(inp, max_length=max_len, do_sample=False, num_beams=4)[0]["generated_text"]
    return result.strip()

def advanced_correct(text: str) -> str:
    # 1) fast spelling
    stage1 = quick_spell_fix(text)
    # 2) transformer grammar/context
    stage2 = grammar_correct(stage1)
    return stage2

def main():
    print(f"Advanced Autocorrector (model: {active_model})")
    print("Type text to correct. Press Enter on empty line to exit.\n")
    while True:
        try:
            raw = input("» ")
        except EOFError:
            break
        if not raw.strip():
            break
        corrected = advanced_correct(raw)
        print("✓", corrected, "\n")

if __name__ == "__main__":
    main()