from collections import Counter
import re

from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()


class Rouge(object):
    def __init__(self):
        self.N = 2
        self.stem = True

    @staticmethod
    def _format_sentence(sentence):
        s = sentence.lower()
        s = re.sub(r"[^0-9a-z]", " ", s)
        s = re.sub(r"\s+", " ", s)
        s = s.strip()
        return s

    @staticmethod
    def _create_n_gram(raw_sentence, n, stem=False):
        res = {}
        sentence = Rouge._format_sentence(raw_sentence)
        tokens = sentence.split(' ')
        if stem:
            # try:  # TODO older NLTK has a bug in Porter Stemmer
            tokens = [stemmer.stem(t) for t in tokens]
            # except:
            #     pass
        sent_len = len(tokens)
        for _n in range(n):
            buf = Counter()
            for idx, token in enumerate(tokens):
                if idx + _n >= sent_len:
                    break
                ngram = ' '.join(tokens[idx: idx + _n + 1])
                buf[ngram] += 1
            res[_n] = buf
        return res

    def compute_rouge(self, references, systems):
        assert (len(references) == len(systems))

        peer_count = len(references)

        result_buf = {}
        for n in range(self.N):
            result_buf[n] = {'p': 0.0, 'r': 0.0, 'f': 0.0}

        for ref_sent, sys_sent in zip(references, systems):
            ref_ngrams = self._create_n_gram(ref_sent, self.N, self.stem)
            sys_ngrams = self._create_n_gram(sys_sent, self.N, self.stem)
            for n in range(self.N):
                ref_ngram = ref_ngrams[n]
                sys_ngram = sys_ngrams[n]
                ref_count = sum(ref_ngram.values())
                sys_count = sum(sys_ngram.values())
                match_count = 0
                for k, v in sys_ngram.items():
                    if k in ref_ngram:
                        match_count += min(v, ref_ngram[k])
                p = match_count / sys_count if sys_count != 0 else 0
                r = match_count / ref_count if ref_count != 0 else 0
                f = 0 if (p == 0 or r == 0) else 2 * p * r / (p + r)
                result_buf[n]['p'] += p
                result_buf[n]['r'] += r
                result_buf[n]['f'] += f

        for n in range(self.N):
            result_buf[n]['p'] /= peer_count
            result_buf[n]['r'] /= peer_count
            result_buf[n]['f'] /= peer_count

        return result_buf
