import argparse
import os

from openai import OpenAI
from tqdm import tqdm

import data
import prompts

from reform import gec_reform_output, gec_reform_output_nltk


def main(args):
    client = OpenAI(
        api_key="sk-...",   # your API key
    )

    temperature = args.temperature
    shot = args.shot
    selection = args.selection
    test_data = args.test_data
    
    train_src_fn = '../data/wi+locness/train.src'
    train_trg_fn = '../data/wi+locness/train.tgt'
    demo_idx_fn = '../data/' + test_data + '/index/' + selection + '.txt'
    test_src_fn = '../data/' + test_data + '/test.src'
    original_output_fn = '../output/' + test_data + '.' + selection + '.' + str(shot) + '.gpt.out'
    output_fn = '../output/' + test_data + '.' + selection + '.' + str(shot) + '.gpt.tgt'

    srcs = data.read_lines(test_src_fn)
    train_demos = data.read_train_gec(train_src_fn, train_trg_fn)
    demo_indexes = data.read_demo_index(demo_idx_fn, shot=shot)
    message_list = []

    print('Preparing prompts...')
    for i in tqdm(range(0, len(srcs)), ncols=60):
        src = srcs[i]
        demo_index = demo_indexes[i]
        demos = [train_demos[idx] for idx in demo_index]
        messages = prompts.gec_few_shot_openai(src, demos)
        message_list.append(messages)

    print('Generating responses...')
    with open(original_output_fn, mode='w', encoding='utf8') as origin_out:
        for i in tqdm(range(len(message_list)), ncols=60):
            messages = message_list[i]
            while True:
                try:
                    response = client.chat.completions.create(
                        messages=messages,
                        model="gpt-3.5-turbo",
                        temperature=temperature,
                    )
                except:
                    print("\nopenai.error.ServiceUnavailableError")
                    continue
                else:
                    break

            tgt = response.choices[0].message.content
            origin_out.write(f'###{i}###' + '\n')
            origin_out.write(tgt + '\n')
    
    origin_out.close()
    
    print('Reforming responses...')
    with open(original_output_fn, 'r', encoding='utf8') as origin_in:
        with open(output_fn, 'w', encoding='utf8') as f_out:
            lines = origin_in.readlines()
            answers = []
            answer = ''
            cnt = 0
            for i in range(len(lines)):
                if lines[i].startswith('###'):
                    if cnt > 0:
                        answers.append(answer)
                    cnt += 1
                    answer = ''
                else:
                    answer += lines[i]
            answers.append(answer)

            for i in tqdm(range(len(srcs)), ncols=60):
                src = srcs[i]
                answer = answers[i]
                if args.test_data == 'conll14':
                    tgt = gec_reform_output_nltk(src, answer)
                else:
                    tgt = gec_reform_output(src, answer)
                f_out.write(tgt + '\n')
        f_out.close()
    origin_in.close()

    os.remove(original_output_fn)
    
    print('Done.')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--selection', type=str, required=True)
    argparser.add_argument('--temperature', type=float, default=0.0)
    argparser.add_argument('--shot', type=int, default=4)
    argparser.add_argument('--test_data', type=str, default='bea19')
    args = argparser.parse_args()
    main(args)