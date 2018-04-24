from collections import Counter
import re

from nltk.stem.porter import PorterStemmer
import numpy as np, scipy.stats as st

stemmer = PorterStemmer()


class Rouge(object):
    def __init__(self, stem=True, use_ngram_buf=False):
        self.N = 2
        self.stem = stem
        self.use_ngram_buf = use_ngram_buf
        self.ngram_buf = {}

    @staticmethod
    def _format_sentence(sentence):
        s = sentence.lower()
        s = re.sub(r"[^0-9a-z]", " ", s)
        s = re.sub(r"\s+", " ", s)
        s = s.strip()
        return s

    def _create_n_gram(self, raw_sentence, n, stem):
        if self.use_ngram_buf:
            if raw_sentence in self.ngram_buf:
                return self.ngram_buf[raw_sentence]
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
        if self.use_ngram_buf:
            self.ngram_buf[raw_sentence] = res
        return res

    def get_ngram(self, sents, N, stem=False):
        if isinstance(sents, list):
            res = {}
            for _n in range(N):
                res[_n] = Counter()
            for sent in sents:
                ngrams = self._create_n_gram(sent, N, stem)
                for this_n, counter in ngrams.items():
                    # res[this_n] = res[this_n] + counter
                    self_counter = res[this_n]
                    for elem, count in counter.items():
                        if elem not in self_counter:
                            self_counter[elem] = count
                        else:
                            self_counter[elem] += count
            return res
        elif isinstance(sents, str):
            return self._create_n_gram(sents, N, stem)
        else:
            raise ValueError

    def get_mean_sd_internal(self, x):
        mean = np.mean(x)
        sd = st.sem(x)
        res = st.t.interval(0.95, len(x) - 1, loc=mean, scale=sd)
        return (mean, sd, res)

    def compute_rouge(self, references, systems):
        assert (len(references) == len(systems))

        peer_count = len(references)

        result_buf = {}
        for n in range(self.N):
            result_buf[n] = {'p': [], 'r': [], 'f': []}

        for ref_sent, sys_sent in zip(references, systems):
            ref_ngrams = self.get_ngram(ref_sent, self.N, self.stem)
            sys_ngrams = self.get_ngram(sys_sent, self.N, self.stem)
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
                result_buf[n]['p'].append(p)
                result_buf[n]['r'].append(r)
                result_buf[n]['f'].append(f)

        res = {}
        for n in range(self.N):
            n_key = 'rouge-{0}'.format(n + 1)
            res[n_key] = {}
            if len(result_buf[n]['p']) >= 50:
                res[n_key]['p'] = self.get_mean_sd_internal(result_buf[n]['p'])
                res[n_key]['r'] = self.get_mean_sd_internal(result_buf[n]['r'])
                res[n_key]['f'] = self.get_mean_sd_internal(result_buf[n]['f'])
            else:
                # not enough samples to calculate confidence interval
                res[n_key]['p'] = (np.mean(np.array(result_buf[n]['p'])), 0, (0, 0))
                res[n_key]['r'] = (np.mean(np.array(result_buf[n]['r'])), 0, (0, 0))
                res[n_key]['f'] = (np.mean(np.array(result_buf[n]['f'])), 0, (0, 0))

        return res
