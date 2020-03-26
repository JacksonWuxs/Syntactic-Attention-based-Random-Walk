from re import compile as re_compile

transfer_table = [
    ("it's", 'it is'), ("i'm", 'i am'), ("he's", 'he is'), ("she's", 'she is'),
    ("we're", 'we are'), ("they're", "they are"), ("you're", 'you are'), ("that's", 'that is'),
    ("this's", 'this is'), ("can't", 'can not'), ("don't", 'do not'), ("doesn't", 'does not'),
    ("we've", 'we have'), ("i've", 'i have'), ("isn't", 'is not'), ("won't", 'will not'),
    ("didn't", 'did not'), ("hadn't", 'had not'), ("what's", 'what is'), ("couldn't", 'clould not'),
    ("you'll", 'you will'), ("you've", 'you have')
]
def replace_abbreviations(text):
    for abbr, sep in transfer_table:
        text = text.replace(abbr, sep)
    return text

from string import punctuation
BLANK = re_compile(u'[ |\\n]{2,}')
WEB_PATTERN = re_compile('(ht|f)tp(s?)\:\/\/[0-9a-zA-Z]([-.\w]*[0-9a-zA-Z])*(:(0-9)*)*(\/?)([a-zA-Z0-9\-\.\?\,\'\/\\\+&amp;%\$#_]*)?')
PUNCTUATION = str.maketrans('', '', punctuation)
NUM_PATTERN = re_compile('[-]{0,1}([1-9]{0,1}[0-9]{0,}[.]{0,1}[0-9]+?|[0-9]{0,3}[,][0-9]{0,3}[,][0-9]{0,3})$')
def clean_text(text):
    text = BLANK.sub(' ', text).strip().replace('.')
    text = replace_abbreviations(text)
    text = WEB_PATTERN.sub('<web>', text)
    text = NUM_PATTERN.sub('<num>', text)
    return text.translate(PUNCTUATION)

def load_spacy_parser(path='en_core_web_md'):
    from spacy import load as spacy_load
    PARSER = spacy_load(path)
    PARSER.remove_pipe('ner')
    return lambda text: PARSER(clean_text(text))

