def read_lines(fn):
    test_srcs = []
    with open(fn, 'r', encoding='utf8') as f_test:
        for line in f_test:
            test_src = line.strip()
            test_srcs.append(test_src)
    return test_srcs


def read_train_gec(src_fn, trg_fn):
    train_paras = []
    with open(src_fn, 'r', encoding='utf8') as f_src, open(trg_fn, 'r', encoding='utf8') as f_trg:
        for src, trg in zip(f_src, f_trg):
            src = src.strip()
            trg = trg.strip()
            train_paras.append((src, trg))
    return train_paras


def read_demo_index(fn, shot=-1):
    demo_indexes = []
    with open(fn, 'r', encoding='utf8') as f_demo:
        for line in f_demo:
            line = line.strip().split(' ')
            demo_index = [int(i) for i in line]
            if shot >= 0:
                demo_index = demo_index[:shot]
            demo_indexes.append(demo_index)
    return demo_indexes
