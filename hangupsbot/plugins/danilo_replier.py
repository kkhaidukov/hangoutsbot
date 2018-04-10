from collections import deque
import os
import logging
import random
from urllib.parse import quote_plus

import itertools
import nltk
from nltk.stem.snowball import RussianStemmer
from nltk import pos_tag, word_tokenize
import requests

MESSAGES_FILE = '%s/danilo_messages.txt' % os.path.dirname(os.path.realpath(__file__))
SENTENCE_LENGTH_LIMIT = 10
PREFIX_LENGTH_LIMIT = 5
ALPHABET = 'йцукенгшщзххъёфывапролджэячсмитьбю'
LINE_START = '~'
IMAGE_EXTENSIONS = ['jpg', 'jpeg', 'png', 'gif']
DIALOG_CONTINUATION_PROBABILITY = 0.2

logger = logging.getLogger(__name__)
rs = RussianStemmer()


class Replier(object):
    def __init__(self):
        # dict of {word: [list of messages with this word]}
        self.danilo_messages = {}
        # dict of {(one, two): [list of threes]}
        self.trigrams_dict = {}
        # {stem: (LINE_START, full_word),}
        self.initials = {}
        self.inverted_trigrams = {}
        self.trigrams = []
        self.current_dialog_length = 0

        self.init_messages()
        # self.init_ngrams()
        self.init_ngrams_no_over_line_trigrams()
        self.init_initials()
        self.init_inverted_trigrams()

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

    def dot_to_line_start(self, token):
        if token == '.':
            return LINE_START
        else:
            return token

    def init_ngrams_no_over_line_trigrams(self):
        logger.info("INIT_NGRAMS_NO_OVER_LINE_TRIGRAMS")
        self.trigrams = []
        # raw = ''
        with open(MESSAGES_FILE) as f:
            for line in f.readlines():
                line_ = '%s %s' % (LINE_START, line)

                tokens = nltk.word_tokenize(line_)
                line_trigrams = list(nltk.trigrams(tokens))
                self.trigrams += line_trigrams
                for one, two, three in line_trigrams:
                    one, two, three = map(self.dot_to_line_start, (one, two, three))
                    self.trigrams_dict.setdefault((one, two), []).append(three)
        logger.info('len(self.trigrams_dict.keys()) = %s' % len(self.trigrams_dict.keys()))

    def init_ngrams(self):
        logger.info("INIT_NGRAMS")
        raw = ''
        with open(MESSAGES_FILE) as f:
            for line in f.readlines():
                raw += '%s %s\n' % (LINE_START, line)

        tokens = nltk.word_tokenize(raw)
        self.trigrams = list(nltk.trigrams(tokens))
        for one, two, three in self.trigrams:
            one, two, three = map(self.dot_to_line_start, (one, two, three))
            self.trigrams_dict.setdefault((one, two), []).append(three)
        logger.info('len(self.trigrams_dict.keys()) = %s' % len(self.trigrams_dict.keys()))

    def init_initials(self):
        logger.info("INIT_INITIALS")
        for k in self.trigrams_dict.keys():
            if k[0] == LINE_START:
                self.initials[rs.stem(k[1])] = k
        logger.info('len(self.initials) = %s' % len(self.initials))

    def init_inverted_trigrams(self):
        """
        dict of inverted trigrams to be used for prepending
        turns a trigram (a, b, c) into a dict of {(b, c): a,}
        """
        logger.info("INIT_INVERTED_TRIGRAMS")
        for a, b, c in self.trigrams:
            self.inverted_trigrams.setdefault((b, c), []).append(a)
        logger.info('len(self.inverted_trigrams) = %s' % len(self.inverted_trigrams))

    def _has_image_link(self, message):
        if 'http' not in message:
            return False
        for ext in IMAGE_EXTENSIONS:
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

    def should_continue_dialog(self):
        logger.info(
            'Determining if a the dialog should continue, self.current_dialog_length = %s' % self.current_dialog_length)
        # in general, it should not start a dialog based on this randomization
        if self.current_dialog_length == 0:
            return False

        # the probability of a consecutive response will diminish with every other message
        range_ = 10
        return random.randrange(range_) < range_ * DIALOG_CONTINUATION_PROBABILITY - self.current_dialog_length

    def get_response(self, message):
        # determine if it even wants to continue the conversation
        will_continue_dialog = self.should_continue_dialog()
        logger.info('will_continue_dialog = %s' % will_continue_dialog)

        response = None
        if self._has_image_link(message):
            response = self.get_response_to_an_image(message)
        # fairly rarely try to be relevant
        elif will_continue_dialog or random.randrange(10) > 8:
            # but if it's not relevant then we'll drop a general generated message
            response = self.generate_message_with_pos_tagging(message) or self.generate_message()
        # sometimes respond with a picture
        elif will_continue_dialog or random.randrange(10) > 8 or (
                'данило' in message.lower() and 'картинку' in message.lower()):
            message_ = message.lower().replace('данило', '').replace('картинку', '').strip()
            response = self.get_image_url_response(message_)
        elif will_continue_dialog or 'данило' in message.lower():
            if random.choice([0, 1]):
                response = self.generate_message()
            else:
                response = self.generate_message_with_pos_tagging(message) or self.generate_message()
        if response:
            # the probability of another response should decrease with every message added
            self.current_dialog_length += 1

        # don't let it get too low
        if self.current_dialog_length > 5:
            self.current_dialog_length = 0

        return response

    def sentence_appendleft(self, sentence, prefix):
        d = deque(sentence)
        d.appendleft(prefix)
        return list(d)

    def sentence_append(self, i, sentence):
        one, two = sentence[-2:]
        next_list = self.trigrams_dict.get((one, two))
        if not next_list:
            return None
        next_ = random.choice(next_list)
        if not next_:
            return None
        elif i > SENTENCE_LENGTH_LIMIT and next_.strip() in ['!', '?']:
            sentence.append(next_)
            return None
        elif i > SENTENCE_LENGTH_LIMIT and next_.strip() in ['.', ',']:
            return None
        elif next_:
            sentence.append(next_)
            return i + 1
        else:
            return None

    def sentence_prepend(self, sentence):
        logger.info('Prepending the sentence, sentence so far: %s' % sentence)
        prefix_length = 0
        debug_prefix_deque = deque()
        while True:
            logger.info('sentence[:2]: %s' % sentence[:2])
            # logger.info('tuple(sentence[:2]): %s' % tuple(sentence[:2]))
            suffix = tuple(sentence[:2])
            logger.info('Suffix: %s' % str(suffix))
            prefix_list = self.inverted_trigrams.get(suffix)
            if not prefix_list:
                logger.info('Got an empty prefix list for suffix=%s' % str(suffix))
                break
            prefix = random.choice(prefix_list)
            logger.info('Got prefix: %s' % prefix)
            if prefix:
                if prefix == LINE_START and prefix_length > PREFIX_LENGTH_LIMIT:
                    sentence = self.sentence_appendleft(sentence, prefix)
                    debug_prefix_deque.appendleft(prefix)
                    break
                else:
                    sentence = self.sentence_appendleft(sentence, prefix)
                    debug_prefix_deque.appendleft(prefix)
                    prefix_length += 1
            else:
                break
        logger.info('Prepended prefix: %s' % debug_prefix_deque)
        return sentence

    def generate_message(self, initial=None):
        logger.info('Using generate_message to create a response, initial = %s' % str(initial))
        if initial:
            if type(initial) == str:
                sentence = [w.strip().lower() for w in initial.split()]
            else:
                sentence = list(initial)
        else:
            sentence = list(random.choice(list(self.initials.values())))
        i = 0
        while True:
            i = self.sentence_append(i, sentence)
            if not i:
                break

        # flip a coin to prepend
        if random.randrange(2):
            # not an in-place change!
            sentence = self.sentence_prepend(sentence)

        return self.join_sentence(sentence)

    def join_sentence(self, sentence):
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
            logger.info("Could not get S, V from message '%s'" % message)
        logger.info("Could not generate anything from message '%s'" % message)

    def generate_kind_of_relevant_message(self, message):
        logger.info('Using generate_kind_of_relevant_message to respond to "%s"' % message)
        # words = [w.strip() for w in message.split()]
        tagged = pos_tag(word_tokenize(message), lang='rus')
        # find meaningful words
        meaningful_words = []
        for word, tag in tagged:
            if tag in ['S', 'V', 'A=m', 'A=f', 'A=n', 'A-PRO=m', 'A-PRO=f', 'A-PRO=n', 'S-PRO']:
                meaningful_words.append(word)
        if meaningful_words and len(meaningful_words) > 1:
            # generate all pairs
            permutations_ = list(itertools.permutations(meaningful_words, 2))
            random.shuffle(permutations_)
            for pair in permutations_:
                # go through the pairs and try to find a corresponding trigram
                trigram = self.trigrams_dict.get(pair)
                if trigram:
                    # append
                    sentence = list(pair)
                    i = 0
                    while True:
                        i = self.sentence_append(i, sentence)
                        if not i:
                            break
                    # prepend
                    sentence = self.sentence_prepend(sentence)
                    return self.join_sentence(sentence)
            else:
                logger.info("Could not find a trigram from meaningful_words pairs.")
        else:
            logger.info("No or not enough 'meaningful' words found")

    def get_image_url_response(self, message):
        logger.info('Using get_image_url_response to respond to "%s"' % message)
        tagged = pos_tag(word_tokenize(message), lang='rus')
        s_v = [word for word, key in tagged if key in ['S', 'V']]
        random.shuffle(s_v)
        base_url = 'https://api.qwant.com/api/search/images?count=10&offset=1&q=%s'
        while s_v:
            q = quote_plus(' '.join(s_v))
            url = base_url % q
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
            logger.info('Calling image search URL: %s' % url)
            result = requests.get(url, headers=headers).json()
            if result['data']['result']['total']:
                media_urls = [item['media'] for item in result['data']['result']['items']]
                random.shuffle(media_urls)
                for media_url in media_urls:
                    for ext in IMAGE_EXTENSIONS:
                        if media_url.endswith('.%s' % ext):
                            return media_url
                    else:
                        logger.warning('None of the URLs in the search results contained a proper image')
            else:
                logger.info('No results, will retry with a smaller query')
                s_v.pop()
        else:
            logger.info('Could not get anything with given message')


# replier = Replier()
# replier.generate_kind_of_relevant_message('или как вариант на домашний этой тетке, если ты его знаешь, кнешна')
# replier.generate_haiku()
