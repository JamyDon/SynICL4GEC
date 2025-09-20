import nltk
import spacy

get_tag = 'corrected sentence'

def gec_extract_answer(string):
    if '<' + get_tag +'>' not in string or 'No errors found' in string:
        return '<no-correction>'
    if '</' + get_tag + '>' not in string:
        # string += ('</' + get_tag + '>')
        return '<no-correction>'
    answer = string.split('<' + get_tag +'>')[1].split('</' + get_tag +'>')[0]
    answer = answer.strip()

    if answer == '' or answer == ' ':
        return '<no-correction>'

    return answer


def gec_reform_answer(string):
    contractions = ["n't", "'d", "'ll", "'m", "'re", "'s", "'ve"]
    tokens = string.split()
    for i in range(len(tokens)):
        if tokens[i] in contractions:
            tokens[i-1] = tokens[i-1] + tokens[i]
            tokens[i] = ''
    tokens = [token for token in tokens if token != '']
    answer = ' '.join(tokens)
    return answer


def gec_reform_output(src, output):
    nlp = spacy.load("en_core_web_sm")

    if '<' + get_tag +'>' not in output:
        output = '<' + get_tag +'>' + output
    answer = gec_extract_answer(output)

    if answer == '<no-correction>':
        answer = src

    answer = gec_reform_answer(answer)

    doc = nlp(answer)
    tokens = []
    for token in doc:
        tokens.append(token.text)
    tokenized_text = ' '.join(tokens).strip()
    return tokenized_text


def gec_reform_output_nltk(src, output):
    if '<' + get_tag +'>' not in output:
        output = '<' + get_tag +'>' + output
    answer = gec_extract_answer(output)

    if answer == '<no-correction>':
        answer = src

    answer = gec_reform_answer(answer)

    tokens = nltk.word_tokenize(answer)
    tokenized_text = ' '.join(tokens).strip()
    return tokenized_text

