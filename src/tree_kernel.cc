#include<fstream>
#include<iostream>
#include<math.h>
#include<map>
#include<stdio.h>
#include<string>
#include<string.h>
#include<time.h>
#include<unordered_map>
#include<vector>

#define N   1000
#define P   1000

struct Node {
    int idx;
    std::string word;
    std::string label;
    Node *head;
    std::vector<Node*> children;
    std::vector<Node*> all_children;
};

std::vector<Node*> read_tree_file(const char *tree_fn);
std::vector<int*> read_idx_file(const char *fn);
void read_split_line(std::string, int*);
void read_split_line(std::string, float*);
int get_size(Node* root);
bool is_leaf(Node* node);
bool is_err(std::string label);
std::vector<Node*> get_all_children(Node* root);
float comp_sim(Node* root1, Node* root2);

std::vector<Node*> read_tree_file(const char *tree_fn) {
    std::ifstream tfs(tree_fn);
    std::string line;
    std::vector<Node*> roots;
    int cnt = 0;

    while(true) {
        std::vector<Node*> nodes;
        std::vector<int> head_idxs;

        nodes.push_back(new Node());
        nodes[0]->idx = 0;
        nodes[0]->word = "<ROOT>";
        nodes[0]->label = "NIL";
        nodes[0]->head = NULL;
        head_idxs.push_back(-1);

        while (std::getline(tfs, line)) {
            if (line == "")
                break;

            Node* node = new Node();
            char tmp[2];
            char word[128];
            char label[128];
            int head_idx;

            sscanf(line.c_str(), "%d\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s", 
                &node->idx, word, tmp, tmp, tmp, tmp, &head_idx, label, tmp, tmp);

            node->word = word;
            node->label = label;

            nodes.push_back(node);
            head_idxs.push_back(head_idx);
        }

        if (nodes.size() <= 1)  break;

        for (int i = 1; i < nodes.size(); ++i) {
            nodes[i]->head = nodes[head_idxs[i]];
            nodes[head_idxs[i]]->children.push_back(nodes[i]);
        }

        for (int i = 0; i < nodes.size(); ++i)
            nodes[i]->all_children = get_all_children(nodes[i]);

        roots.push_back(nodes[0]);

        std::cout<< "Reading " << ++cnt << '\r';
        fflush(stdout);
    }

    std::cout << std::endl;

    return roots;
}

std::vector<int*> read_idx_file(const char *fn) {
    std::ifstream ifs(fn);
    std::string line;
    std::vector<int*> idxs;

    while (std::getline(ifs, line)) {
        if (line == "") break;
        int *idx = new int[P];
        read_split_line(line, idx);
        idxs.push_back(idx);
    }

    return idxs;
}

void read_split_line(std::string line, int* idx) {
    int i = 0;
    int j = 0;
    while (i < line.length()) {
        int k = i;
        while (k < line.length() && line[k] != ' ') ++k;
        std::string tmp = line.substr(i, k - i);
        idx[j++] = std::stoi(tmp);
        i = k + 1;
    }
}

void read_split_line(std::string line, float* vec) {
    int i = 0;
    int j = 0;
    while (i < line.length()) {
        int k = i;
        while (k < line.length() && line[k] != ' ') ++k;
        std::string tmp = line.substr(i, k - i);
        vec[j++] = std::stof(tmp);
        i = k + 1;
    }
}

int get_size(Node* root) {
    int size = 1 + root->all_children.size();
    return size;
}

bool is_leaf(Node* node) {
    return node->children.size() == 0;
}

bool is_err(std::string label) {
    return label == "R" || label == "M" || label == "S";
}

std::vector<Node*> get_all_children(Node* root) {
    std::vector<Node*> children;
    for (int i = 0; i < root->children.size(); ++i) {
        children.push_back(root->children[i]);
        std::vector<Node*> tmp = get_all_children(root->children[i]);
        children.insert(children.end(), tmp.begin(), tmp.end());
    }
    return children;
}

float comp_sim(Node* root1, Node* root2) {
    float K = 0;

    for (auto child1 : root1->all_children) {
        for (auto child2 : root2->all_children) {
            if (is_leaf(child1) && is_leaf(child2)) {
                if (child1->label == child2->label) {
                    float score = 1;
                    K += score;
                }
            } else if (!is_leaf(child1) && !is_leaf(child2)) {
                float score = comp_sim(child1, child2);
                K += score;
            }
        }
    }
    
    int M = get_size(root1) * get_size(root2);
    K /= float(M);

    return K;
}

int main(int argc, char *argv[]) {
    clock_t start = clock();

    char *output_fn, *test_tfn, *train_tfn, *idx_fn;
    if (argc == 5) {
        output_fn = argv[1];
        test_tfn = argv[2];
        train_tfn = argv[3];
        idx_fn = argv[4];
    } else if (argc == 4) {
        output_fn = argv[1];
        test_tfn = argv[2];
        train_tfn = argv[3];
        idx_fn = NULL;
    } else {
        std::cout << "Usage: ./tree_kernel output_fn test_tfn train_tfn [idx_fn]" << std::endl;
        return 1;
    }

    std::vector<Node*> test_trees = read_tree_file(test_tfn);
    std::vector<Node*> train_trees = read_tree_file(train_tfn);

    std::vector<int*> idxs;
    if (idx_fn != NULL)
        idxs = read_idx_file(idx_fn);

    std::ofstream ofs(output_fn, std::ios::app);
    int test_size = test_trees.size();
    int train_size = train_trees.size();

    char buf[102];
    memset(buf, '\0', sizeof(buf));
    const char *label = "|/-\\";

    for (int i = 0; i < test_size; ++i) {
        printf("[%-101s][%d%%][%c]\r", buf, (int)(100.0 * i / test_size), label[i % 4]);
        fflush(stdout);
        buf[(int)(100.0 * i / test_size)] = '=';

        std::vector<std::pair<int, float> > sims;

        if (idx_fn != NULL) {
            for (int j = 0; j < P; ++j) {
                int idx = idxs[i][j];
                sims.push_back(std::make_pair(idx, comp_sim(test_trees[i], train_trees[idx])));
            }
        } else {
            for (int j = 0; j < train_size; ++j) {
                sims.push_back(std::make_pair(j, comp_sim(test_trees[i], train_trees[j])));
            }
        }

        std::vector<int> idxs;
        for (int j = 0; j < N; ++j) {
            float max_sim = -1;
            int max_idx = 0;
            int clear_idx = 0;
            for (int k = 0; k < sims.size(); ++k) {
                if (sims[k].second > max_sim) {
                    max_sim = sims[k].second;
                    max_idx = sims[k].first;
                    clear_idx = k;
                }
            }
            idxs.push_back(max_idx);
            sims[clear_idx].second = -1;
        }
        
        for (int j = 0; j < N - 1; ++j) {
            ofs << idxs[j] << " ";
        }
        ofs << idxs[N - 1] << std::endl;
    }

    ofs.close();

    clock_t end = clock();
    std::cout << "done" << std::endl;
    std::cout << "Time: " << (double)(end - start) / CLOCKS_PER_SEC << "s" << std::endl;

    return 0;
}