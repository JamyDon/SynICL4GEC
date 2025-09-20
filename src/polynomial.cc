#include<float.h>
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
#define N_LABELS    59
#define N_DIM       2 * N_LABELS
#define X2Y(x)      (x + N_LABELS)
#define WEIGHT      1.0
#define LABEL_FN    "../data/dependency_types.txt"

struct Node {
    int idx;
    std::string word;
    std::string label;
    Node *head;
    std::vector<Node*> children;
    std::vector<Node*> all_children;
};

typedef std::unordered_map<std::string, char> LabelMap;

std::vector<std::vector<int*> > read_tree_file(const char *tree_fn);
std::vector<int*> read_idx_file(const char *fn);
LabelMap read_label_file(const char *label_fn);
void read_split_line(std::string, int*);
int get_size(Node* root);
bool is_leaf(Node* node);
bool is_err(std::string label);
std::vector<int*> tree2polynomial(Node* root, LabelMap &label_map);
void polynomial_mul(std::vector<int*> &poly1, std::vector<int*> &poly2);
std::vector<Node*> get_all_children(Node* root);
float manhattan_dist(int* vec1, int* vec2);
float polynomial_dist(std::vector<int*> poly1, std::vector<int*> poly2);

std::vector<std::vector<int*> > read_tree_file(const char *tree_fn, LabelMap &label_map) {
    std::ifstream tfs(tree_fn);
    std::string line;
    std::vector<std::vector<int*> > polys;
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

        auto poly = tree2polynomial(nodes[0], label_map);
        polys.push_back(poly);

        std::cout<< "Reading " << ++cnt << '\r';
        fflush(stdout);
    }

    std::cout << std::endl;

    return polys;
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

LabelMap read_label_file(const char *label_fn) {
    std::ifstream ifs(label_fn);
    std::string line;
    LabelMap label_map;
    char cnt = 0;

    while (std::getline(ifs, line)) {
        if (line == "") break;
        std::string label;
        char tmp[16];
        sscanf(line.c_str(), "%s", tmp);
        label = tmp;
        label_map[label] = cnt++;
    }

    return label_map;
}

int get_size(Node* root) {
    int size = 1 + root->all_children.size();
    return size;
}

bool is_leaf(Node* node) {
    return node->children.size() == 0;
}

inline bool is_err(std::string label) {
    return label == "R" || label == "M" || label == "S";
}

std::vector<int*> tree2polynomial(Node* root, LabelMap &label_map) {
    std::vector<int*> polynomial;
    if (is_leaf(root)) {
        int *term = new int[N_DIM];
        memset(term, 0, sizeof(term));
        char x = label_map[root->label];
        term[x] = 1;
        polynomial.push_back(term);
    } else {
        bool first_child = true;
        for (auto child : root->children) {
            std::vector<int*> child_polynomial = tree2polynomial(child, label_map);

            if (first_child)
                polynomial = child_polynomial;
            else
                polynomial_mul(polynomial, child_polynomial);

            first_child = false;
        }
        int *term = new int[N_DIM];
        memset(term, 0, sizeof(term));
        char y = X2Y(label_map[root->label]);
        term[y] = 1;
        polynomial.push_back(term);
    }
    return polynomial;
}

// All the terms in poly2 will be multiplied together before being multiplied by poly1.
// This is a little different from Liu et al. (2022).
// In this way, the complexity of the algorithm is nearly linear.
void polynomial_mul(std::vector<int*> &poly1, std::vector<int*> &poly2) {
    for (int i = 0; i < poly1.size(); ++i) {
        for (int j = 0; j < poly2.size(); ++j) {
            for (int k = 0; k < N_DIM; ++k) {
                poly1[i][k] += poly2[j][k];
            }
        }
    }
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

float manhattan_dist(int* vec1, int* vec2) {
    float dist = 0;
    for (int i = 0; i < 3; ++i)
        dist += WEIGHT * fabs(vec1[i] - vec2[i]);
    for (int i = 3; i < N_LABELS; ++i)
        dist += fabs(vec1[i] - vec2[i]);
    for (int i = N_LABELS; i < N_LABELS + 3; ++i)
        dist += WEIGHT * fabs(vec1[i] - vec2[i]);
    for (int i = N_LABELS + 3; i < N_DIM; ++i)
        dist += fabs(vec1[i] - vec2[i]);
    return dist;
}

float polynomial_dist(std::vector<int*> poly1, std::vector<int*> poly2) {
    float dist = 0;
    int size_1 = poly1.size();
    int size_2 = poly2.size();

    for (int i = 0; i < size_1; ++i) {
        float min_dist_1 = FLT_MAX;
        for (int j = 0; j < size_2; ++j) {
            float tmp_dist = manhattan_dist(poly1[i], poly2[j]);
            if (tmp_dist < min_dist_1)
                min_dist_1 = tmp_dist;
        }
        dist += min_dist_1;
    }
    for (int i = 0; i < size_2; ++i) {
        float min_dist_2 = FLT_MAX;
        for (int j = 0; j < size_1; ++j) {
            float tmp_dist = manhattan_dist(poly2[i], poly1[j]);
            if (tmp_dist < min_dist_2)
                min_dist_2 = tmp_dist;
        }
        dist += min_dist_2;
    }

    dist /= (size_1 + size_2);

    return dist;
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
        std::cout << "Usage: ./polynomial output_fn test_tfn train_tfn [idx_fn]" << std::endl;
        return 1;
    }

    LabelMap label_map = read_label_file(LABEL_FN);
    auto test_trees = read_tree_file(test_tfn, label_map);
    auto train_trees = read_tree_file(train_tfn, label_map);

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

        std::vector<std::pair<int, float> > dists;

        if (idx_fn != NULL) {
            for (int j = 0; j < P; ++j) {
                int idx = idxs[i][j];
                dists.push_back(std::make_pair(idx, polynomial_dist(test_trees[i], train_trees[idx])));
            }
        } else {
            for (int j = 0; j < train_size; ++j) {
                dists.push_back(std::make_pair(j, polynomial_dist(test_trees[i], train_trees[j])));
            }
        }

        std::vector<int> idxs;
        for (int j = 0; j < N; ++j) {
            float min_dist = FLT_MAX;
            int min_idx = 0;
            int clear_idx = 0;
            for (int k = 0; k < dists.size(); ++k) {
                if (dists[k].second < min_dist) {
                    min_dist = dists[k].second;
                    min_idx = dists[k].first;
                    clear_idx = k;
                }
            }
            idxs.push_back(min_idx);
            dists[clear_idx].second = FLT_MAX;
        }

        for (int j = 0; j < N - 1; ++j) {
            ofs << idxs[j] << " ";
        }
        ofs << idxs[N - 1] << std::endl;
    }

    ofs.close();

    clock_t end = clock();
    std::cout << "done" << std::endl;
    std::cout << "Time Elapsed: " << (double)(end - start) / CLOCKS_PER_SEC << "s" << std::endl;

    return 0;
}