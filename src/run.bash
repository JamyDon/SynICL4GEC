# Step 1: Random, BM25 and BERT selection.
echo "Step 1: Random, BM25 and BERT selection."

BERT_DIR="google-bert/bert-base-uncased"
DEVICE="cuda:0"

python selection.py --bert_dir $BERT_DIR --device $DEVICE

# Step 2: Tree Kernel and Polynomial selection (1-stage & 2-stage).
# Step 2.1: Tree Kernel selection.
echo "Step 2.1: Tree Kernel selection."
g++ -o ../bin/tree_kernel ./tree_kernel.cc
cd ../bin/
TEST_DATA_LIST=("bea19" "conll14")
for TEST_DATA in "${TEST_DATA_LIST[@]}"
do
    # 1-stage.
    OUTPUT_FN="../data/"$TEST_DATA"/index/treekernel.txt"
    TEST_TFN="../data/"$TEST_DATA"/test.gopar"
    TRAIN_TFN="../data/wi+locness/train.gopar"
    ./tree_kernel $OUTPUT_FN $TEST_TFN $TRAIN_TFN

    # 2-stage (based on BM25 and BERT).
    FIRST_STAGES=("bm25" "bert")
    for FIRST_STAGE in "${FIRST_STAGES[@]}"
    do
        OUTPUT_FN="../data/"$TEST_DATA"/index/"$FIRST_STAGE".treekernel.txt"
        TEST_TFN="../data/"$TEST_DATA"/test.gopar"
        TRAIN_TFN="../data/wi+locness/train.gopar"
        IDX_FN="../data/"$TEST_DATA"/index/"$FIRST_STAGE".txt"
        ./tree_kernel $OUTPUT_FN $TEST_TFN $TRAIN_TFN $IDX_FN
    done
done
cd -

# Step 2.2: Polynomial selection.
echo "Step 2.2: Polynomial selection."
g++ -o ../bin/polynomial ./polynomial.cc
cd ../bin/
TEST_DATA_LIST=("bea19" "conll14")
for TEST_DATA in "${TEST_DATA_LIST[@]}"
do
    # 1-stage.
    OUTPUT_FN="../data/"$TEST_DATA"/index/polynomial.txt"
    TEST_TFN="../data/"$TEST_DATA"/test.gopar"
    TRAIN_TFN="../data/wi+locness/train.gopar"
    ./polynomial $OUTPUT_FN $TEST_TFN $TRAIN_TFN

    # 2-stage (based on BM25 and BERT).
    FIRST_STAGES=("bm25" "bert")
    for FIRST_STAGE in "${FIRST_STAGES[@]}"
    do
        OUTPUT_FN="../data/"$TEST_DATA"/index/"$FIRST_STAGE".polynomial.txt"
        TEST_TFN="../data/"$TEST_DATA"/test.gopar"
        TRAIN_TFN="../data/wi+locness/train.gopar"
        IDX_FN="../data/"$TEST_DATA"/index/"$FIRST_STAGE".txt"
        ./polynomial $OUTPUT_FN $TEST_TFN $TRAIN_TFN $IDX_FN
    done
done
cd -

# Step 3: LLM inference. The output will be saved in '../output'

TEST_DATA_LIST=("bea19" "conll14")
SELECTIONS=("random" "bm25" "bert" "treekernel" "polynomial" "bm25.treekernel" "bm25.polynomial" "bert.treekernel" "bert.polynomial")
SHOT=4

# Step 3.1: LLaMA inference (7b).
echo "Step 3.1: LLaMA inference (7b)."
GPU=0
LLAMA_DIR=meta-llama/Llama-2-7b-chat-hf
for TEST_DATA in "${TEST_DATA_LIST[@]}"
do
    for SELECTION in "${SELECTIONS[@]}"
    do
        CUDA_VISIBLE_DEVICES=$GPU python llama.py --selection $SELECTION --test_data $TEST_DATA --model $LLAMA_DIR --shot $SHOT
    done
done

# Step 3.2: GPT-3.5-turbo inference.
echo "Step 3.2: GPT-3.5-turbo inference."
for TEST_DATA in "${TEST_DATA_LIST[@]}"
do
    for SELECTION in "${SELECTIONS[@]}"
    do
        python oai.py   --selection $SELECTION --test_data $TEST_DATA --shot $SHOT
    done
done

# Done.
echo "Done."