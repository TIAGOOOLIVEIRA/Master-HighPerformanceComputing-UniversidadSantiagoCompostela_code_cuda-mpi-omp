#include <iostream>
#include <vector>
#include <memory>

/*
clang++ -std=c++17 -I/opt/homebrew/include -L/opt/homebrew/lib -ltbb -o simple_unbalanced_tree_generator  simple_unbalanced_tree_generator.cpp

unbalanced tree: inserts nodes without rotations
insertion order: defines shape of the tree
The generateTree() inserts values sequentially
Inorder traversal prints values in sorted order, regardless of tree shape.

#TODO
    create a cmake file to compile in HPC and MacOS
*/

struct Node {
    int value;
    Node* left;
    Node* right;

    Node(int val) : value(val), left(nullptr), right(nullptr) {}
};

class UnbalancedTree {
public:
    Node* root;

    UnbalancedTree() : root(nullptr) {}

    void insert(int value) {
        root = insertRec(root, value);
    }

    void inorder() const {
        inorderRec(root);
        std::cout << std::endl;
    }

private:
    Node* insertRec(Node* node, int value) {
        if (!node)
            return new Node(value);
        if (value < node->value)
            node->left = insertRec(node->left, value);
        else
            node->right = insertRec(node->right, value);
        return node;
    }

    void inorderRec(Node* node) const {
        if (!node) return;
        inorderRec(node->left);
        std::cout << node->value << " ";
        inorderRec(node->right);
    }
};

UnbalancedTree generateTree(const std::vector<int>& values) {
    UnbalancedTree tree;
    for (int val : values) {
        tree.insert(val);
    }
    return tree;
}

int main() {
    std::vector<int> values = {10, 5, 1, 7, 20, 15, 30};
    UnbalancedTree tree = generateTree(values);

    std::cout << "Inorder traversal of the unbalanced tree: ";
    tree.inorder();

    return 0;
}
