import json
import re
import unicodedata

class DictionaryLookup:
    def __init__(self, jsonl_path, target_lang):
        self.lookup = {}
        self.target_lang = target_lang
        # Standardize POS tags to match what you updated in wals.py
        self.POS_MAP = {
            "noun": "noun",
            "verb": "verb",
            "adjective": "adjective",
            "adverb": "",
            "pronoun": "pronoun",
            "preposition": "prepositional phrase",
            "determiner": "determiner",
            "numeral": "numeral",
            "proper noun": "proper noun"
        }
        self.load_dictionary(jsonl_path)

    def normalize(self, text):
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
        return " ".join(text.casefold().split())

    def load_dictionary(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                word = self.normalize(data['word'])
                trans = data.get(self.target_lang)
                if trans:
                    # Get the POS from dict and map it to your WALS tags
                    raw_pos = data.get('pos', 'other').lower()
                    mapped_pos = self.POS_MAP.get(raw_pos, "OTHER")
                    
                    # Store translation and the standardized POS
                    self.lookup[word] = (trans if isinstance(trans, str) else trans[0], mapped_pos)

    def translate_sentence(self, sentence):
        # Tokenize while keeping punctuation
        raw_tokens = re.findall(r"[\w']+|[.,!?;]", sentence)
        norm_tokens = [self.normalize(t) for t in raw_tokens]
        
        i = 0
        tokens_pos = []
        while i < len(norm_tokens):
            match_found = False
            # Greedy longest match (6-gram down to 1-gram)
            for n in range(6, 0, -1):
                if i + n <= len(norm_tokens):
                    phrase = " ".join(norm_tokens[i:i+n])
                    if phrase in self.lookup:
                        target_word, pos = self.lookup[phrase]
                        tokens_pos.append((target_word, pos))
                        i += n
                        match_found = True
                        break
            if not match_found:
                # Keep original English if word isn't in dictionary
                tokens_pos.append((raw_tokens[i], "OTHER"))
                i += 1
        return tokens_pos