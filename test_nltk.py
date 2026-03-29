import nltk

# Force download required resources
# nltk.download('punkt')
# nltk.download('punkt_tab')

# from nltk.tokenize import word_tokenize

# sentence = "The system should be fast"
# print(word_tokenize(sentence))

import nltk

# Force required downloads (new NLTK + Python 3.13 fix)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

from nltk.tokenize import word_tokenize
from nltk import pos_tag

sentence = "The system should be fast"

tokens = word_tokenize(sentence)
pos_tags = pos_tag(tokens)

print(pos_tags)
