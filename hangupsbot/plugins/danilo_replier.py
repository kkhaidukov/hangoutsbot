import os
import logging
import random
from urllib.parse import quote_plus

import nltk
from nltk.stem.snowball import RussianStemmer
from nltk import pos_tag, word_tokenize
import requests

MESSAGES_FILE = '%s/danilo_messages.txt' % os.path.dirname(os.path.realpath(__file__))
SENTENCE_LENGTH_LIMIT = 20
ALPHABET = 'йцукенгшщзххъёфывапролджэячсмитьбю'
LINE_START = '~'

logger = logging.getLogger(__name__)
rs = RussianStemmer()


class Replier(object):
    def __init__(self):
        self.danilo_messages = {}
        self.trigrams_dict = {}
        # {stem: (LINE_START, full_word),}
        self.initials = {}

        self.init_messages()
        self.init_ngrams()
        self.init_initials()

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

    def init_initials(self):
        for k in self.trigrams_dict.keys():
            if k[0] == LINE_START:
                self.initials[rs.stem(k[1])] = k
        logger.info('len(self.initials) = %s' % len(self.initials))

    def _has_image_link(self, message):
        if 'http' not in message:
            return False
        for ext in ['jpg', 'png', 'gif']:
            if ext in message:
                return True
        else:
            return False

    def get_message_containing_any_word(self, message):
        # will return a non-generated message containing one of the words in `message`
        message_words = [w.strip() for w in message.strip().split()]

        random.shuffle(message_words)
        for w in message_words:
            if w in self.danilo_messages:
                return random.choice(self.danilo_messages[w]).strip()

    def get_response_to_an_image(self, message):
        logger.info('Using get_response_to_an_image to respond to "%s"' % message)
        # oh well
        message = random.choice(['картинка', 'пикча'])
        relevant_message = self.get_message_containing_any_word(message)
        if relevant_message:
            return self.generate_message(relevant_message)

    def get_response(self, message):
        if self._has_image_link(message):
            return self.get_response_to_an_image(message)
        # fairly rarely try to be relevant
        elif random.randrange(10) > 8:
            return self.generate_message_with_pos_tagging(message)
        # sometimes respond with a picture
        elif random.randrange(10) > 8:
            return self.get_image_url_response(message)
        elif 'данило' in message.lower():
            return self.generate_message()

    def generate_message(self, initial=None):
        logger.info('Using generate_message to create a response.')
        if initial:
            if type(initial) == str:
                sentence = [w.strip().lower() for w in initial.split()]
            else:
                sentence = list(initial)
        else:
            sentence = list(random.choice(list(self.initials.values())))
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

    def generate_message_with_pos_tagging(self, message):
        logger.info('Using generate_message_with_pos_tagging to respond to "%s"' % message)
        tagged = pos_tag(word_tokenize(message), lang='rus')
        # find Ss and Vs and stem them
        s_v = []
        for word, tag in tagged:
            if tag in ['S', 'V']:
                s_v.append(rs.stem(word))
        if s_v:
            random.shuffle(s_v)
            # try to find and initial with one of them
            for word in s_v:
                initial = self.initials.get(word)
                # if an initial is found, generate a sentence from there
                if initial:
                    return self.generate_message(initial)
        else:
            logger.info("Could not generate anything from message '%s'" % message)

    def get_image_url_response(self, message):
        logger.info('Using get_image_url_response to respond to "%s"' % message)
        tagged = pos_tag(word_tokenize(message), lang='rus')
        s_v = [word for word, key in tagged if key in ['S', 'V']]
        random.shuffle(s_v)
        base_url = 'https://api.qwant.com/api/search/images?count=1&offset=1&q=%s'
        while s_v:
            q = quote_plus(' '.join(s_v))
            url = base_url % q
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
            logger.info('Calling image search URL: %s' % url)
            result = requests.get(url, headers=headers).json()
            if result['data']['result']['total']:
                return result['data']['result']['items'][0]['media']
            else:
                logger.info('No results, will retry with a smaller query')
                s_v.pop()
        else:
            logger.info('Could not get anything with given message')
