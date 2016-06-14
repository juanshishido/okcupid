import string

from spacy.en import English


nlp = English(tagger=False, parser=False, entity=False)

def spacy_tokenize(text):
    """Use spaCy's default tokenizer, which can
    handle "emoticons and other web-based features,"
    and remove punctuation from the results
    
    Parameters
    ----------
    text : str
        The text to tokenize
        
    Returns
    -------
    list of tokens (as strings)
    
    Notes
    -----
    Source of above quote:
        https://nicschrading.com/project/Intro-to-NLP-with-spaCy/
    """
    assert isinstance(text, str)
    tokens = nlp(text)
    return [str(t) for t in tokens if str(t) not in string.punctuation]
