import os
import logging
import random

import nltk

MESSAGES_FILE = '%s/danilo_messages.txt' % os.path.dirname(os.path.realpath(__file__))
SENTENCE_LENGTH_LIMIT = 20
ALPHABET = 'йцукенгшщзххъёфывапролджэячсмитьбю'
LINE_START = '~'

logger = logging.getLogger(__name__)


class Replier(object):
    def __init__(self):
        self.danilo_messages = {}
        self.trigrams_dict = {}
        self.initials = []

    def clean_word(self, word):
        return ''.join(ch for ch in word if ch.isalnum() or ch == u'-')

    def init_messages(self):
        logger.info("INIT_MESSAGES")
        with open(MESSAGES_FILE) as f:
            for l in f.readlines():
                words = [w.strip() for w in l.split()]
                for w in words:
                    key = self.clean_word(w)
                    self.danilo_messages.setdefault(key, []).append(l)
        logger.info('len(self.danilo_messages.keys()) = %s' % len(self.danilo_messages.keys()))

    def init_ngrams(self):
        logger.info("INIT_NGRAMS")
        raw = ''
        with open(MESSAGES_FILE) as f:
            for line in f.readlines():
                raw += '%s %s\n' % (LINE_START, line)

        tokens = nltk.word_tokenize(raw)
        trigrams = nltk.trigrams(tokens)
        for tr in trigrams:
            self.trigrams_dict.setdefault((tr[0], tr[1]), []).append(tr[2])
        logger.info('len(self.trigrams_dict.keys()) = %s' % len(self.trigrams_dict.keys()))
        for k in self.trigrams_dict.keys():
            if k[0] == LINE_START:
                self.initials.append(k)
        logger.info('len(self.initials) = %s' % len(self.initials))

    def _has_image_link(self, message):
        if 'http' not in message:
            return False
        for ext in ['jpg', 'png', 'gif']:
            if ext in message:
                return True
        else:
            return False

    def get_relevant_message(self, message):
        # will return a non-generated message containing one of the words in `message`
        message_words = [w.strip() for w in message.strip().split()]

        random.shuffle(message_words)
        for w in message_words:
            if w in self.danilo_messages:
                return random.choice(self.danilo_messages[w]).strip()

    def get_response(self, message):
        if not self.danilo_messages:
            self.init_messages()

        if self._has_image_link(message):
            # oh well
            message = random.choice(['картинка', 'пикча'])
            relevant_message = self.get_relevant_message(message)
            if relevant_message:
                return self.generate_message(relevant_message)
        elif 'данило' in message.lower():
            return self.generate_message()

    def generate_message(self, initial=None):
        if initial:
            sentence = [w.strip().lower() for w in initial.split()]
        else:
            sentence = list(random.choice(self.initials))
        i = 0
        while True:
            one, two = sentence[-2:]
            next_ = random.choice(self.trigrams_dict.get((one, two)))
            if not next_:
                break
            elif i > SENTENCE_LENGTH_LIMIT and next_.strip() in ['.', '!', '?']:
                sentence.append(next_)
                break
            elif i > SENTENCE_LENGTH_LIMIT and next_.strip() == ',':
                break
            elif next_:
                sentence.append(next_)
                i += 1
            else:
                break

        sentence_string = ''
        for w in sentence:
            w_ = w.strip().lower()
            if w_ == LINE_START:
                if not sentence_string:
                    pass
                else:
                    sentence_string += '.'
            elif w_ == '-':
                sentence_string += ' - '
            elif len(w_) == 1 and w_ not in ALPHABET:
                sentence_string += w_
            else:
                sentence_string += ' %s' % w_

        return sentence_string.strip()
