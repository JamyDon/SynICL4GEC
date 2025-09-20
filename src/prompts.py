from utils import add_message


def gec_zero_shot(src):
    instruction = 'There is an erroneous sentence between `<erroneous sentence>` and `</erroneous sentence>`. Then grammatical errors in the erroneous sentence will be corrected. The corrected version will be between `<corrected sentence>` and `</corrected sentence>`.'
    src_prompt = '<erroneous sentence> ' + src + ' </erroneous sentence>'
    trg_prompt = '<corrected sentence> '
    prompt = [instruction, src_prompt, trg_prompt]
    prompt = '\n'.join(prompt)
    return prompt


def gec_few_shot(src, demos):
    instruction = 'There is an erroneous sentence between `<erroneous sentence>` and `</erroneous sentence>`. Then grammatical errors in the erroneous sentence will be corrected. The corrected version will be between `<corrected sentence>` and `</corrected sentence>`.'

    prompt = [instruction]

    for demo in demos:
        src_prompt = '<erroneous sentence> ' + demo[0] + ' </erroneous sentence>'
        trg_prompt = '<corrected sentence> ' + demo[1] + ' </corrected sentence>'
        prompt.append(src_prompt)
        prompt.append(trg_prompt)

    src_prompt = '<erroneous sentence> ' + src + ' </erroneous sentence>'
    trg_prompt = '<corrected sentence> '
    prompt.append(src_prompt)
    prompt.append(trg_prompt)
    
    prompt = '\n'.join(prompt)
    return prompt


def gec_few_shot_openai(src, demos):
    messages = []

    instruction = 'You are a grammar correction assistant. The user will give you a sentence with grammatical errors (between `<erroneous sentence>` and `</erroneous sentence>`). You need to correct the sentence (between `<corrected sentence>` and `</corrected sentence>`). Requirements: 1. Make as few changes as possible. 2. Make sure the sentence has the same meaning as the original sentence. 3. If there is no error, just output `No errors found`. '

    add_message(messages, 'system', instruction)

    for demo in demos:
        src_prompt = '<erroneous sentence> ' + demo[0] + ' </erroneous sentence>'
        trg_prompt = '<corrected sentence> ' + demo[1] + ' </corrected sentence>'
        add_message(messages, 'user', src_prompt)
        add_message(messages, 'assistant', trg_prompt)

    src_prompt = '<erroneous sentence> ' + src + ' </erroneous sentence>'
    add_message(messages, 'user', src_prompt)

    return messages
