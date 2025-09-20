import argparse
import torch
import transformers

from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.pipelines.pt_utils import KeyDataset

import data
import prompts

from reform import gec_reform_output, gec_reform_output_nltk


# in order to fit into the GPU memory
batch_size_dict = {1: 16, 2: 8, 4: 8, 8: 4, 16: 2}


def main(args):
    shot = args.shot
    selection = args.selection
    batch_size = batch_size_dict[shot]
    max_new_tokens = args.max_new_tokens

    model =  args.model

    test_data = args.test_data
    train_src_fn = '../data/wi+locness/train.src'
    train_trg_fn = '../data/wi+locness/train.tgt'
    demo_idx_fn = '../data/' + test_data + '/index/' + selection + '.txt'
    test_src_fn = '../data/' + test_data + '/test.src'
    output_fn = '../output/' + test_data + '.' + selection + '.' + str(shot) + '.' + 'llama.tgt'

    tokenizer = AutoTokenizer.from_pretrained(model)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device_map='auto',
    )
    pipeline.tokenizer.pad_token_id = tokenizer.eos_token_id

    srcs = data.read_lines(test_src_fn)
    train_demos = data.read_train_gec(train_src_fn, train_trg_fn)
    demo_indexes = data.read_demo_index(demo_idx_fn, shot=shot)
    prompt_list = {'prompts': []}
    responses = []

    print('Preparing prompts...')
    for i in tqdm(range(len(srcs)), ncols=60):
        src = srcs[i]
        demo_index = demo_indexes[i]
        demos = [train_demos[idx] for idx in demo_index]
        prompt = prompts.gec_few_shot(src, demos)
        prompt_list['prompts'].append(prompt)
    
    dataset = Dataset.from_dict(prompt_list)

    print('Generating responses...')
    with torch.no_grad():
        for output in tqdm(pipeline(
            KeyDataset(dataset, 'prompts'),
            do_sample=False,
            return_full_text=False,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
        ), ncols=60):
            response = output[0]['generated_text']
            response = '<corrected sentence> ' + response
            responses.append(response)
            # torch.cuda.empty_cache()
    
    print('Reforming responses...')
    with open(output_fn, 'w', encoding='utf8') as f_out:
        for i in tqdm(range(len(responses)), ncols=60):
            src = srcs[i]
            response = responses[i]
            if args.test_data == 'conll14':
                trg = gec_reform_output_nltk(src, response)
            else:
                trg = gec_reform_output(src, response)
            f_out.write(trg + '\n')
    f_out.close()

    print('Done.')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--selection', type=str, required=True)
    argparser.add_argument('--shot', type=int, default=4)
    argparser.add_argument('--batch_size', type=int, default=1)
    argparser.add_argument('--model', type=str, required=True)
    argparser.add_argument('--test_data', type=str, required=True)
    argparser.add_argument('--max_new_tokens', type=int, default=64)
    args = argparser.parse_args()
    main(args)