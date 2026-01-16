import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# 1. Load text corpus (sample text, you can replace with large dataset)
text = """
This is a simple corpus for building an autocorrect system.
It contains words like spelling, correction, python, language, example, and small.
We will test misspelled words like speling, exampl, langauge and pythno.
"""

# 2. Tokenize words (lowercase, only alphabetic)
words = re.findall(r'\w+', text.lower())
word_freq = Counter(words)

# Convert to DataFrame for easy analysis
df = pd.DataFrame(word_freq.items(), columns=['word', 'count']).sort_values(by="count", ascending=False)

# 3. Candidate Generation (edit distance)
alphabet = 'abcdefghijklmnopqrstuvwxyz'

def edits1(word):
    """All edits that are one edit away"""
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in alphabet]
    inserts = [L + c + R for L, R in splits for c in alphabet]
    return set(deletes + transposes + replaces + inserts)

def known(words_set):
    """Return the subset of words that appear in dictionary"""
    return set(w for w in words_set if w in word_freq)

def candidates(word):
    """Generate possible corrections"""
    return (known([word]) or known(edits1(word)) or [word])

def autocorrect(word):
    """Return most probable correction"""
    return max(candidates(word), key=lambda w: word_freq[w])

# 4. Test the autocorrect system
test_words = ["speling", "exampl", "langauge", "pythno", "smal"]
for w in test_words:
    print(f"{w:10} -> {autocorrect(w)}")

# 5. Visualization of word frequencies
plt.figure(figsize=(8,5))
sns.barplot(x="word", y="count", data=df.head(8), palette="viridis")
plt.title("Top Word Frequencies in Corpus")
plt.show()

