#include <iostream>


/*
clang++ -std=c++17 -I/opt/homebrew/include -L/opt/homebrew/lib -ltbb -o fig_1_04  fig_1_04.cpp

*/

struct Node {
    int data;
    Node* left;
    Node* right;

    Node(int val) : data(val), left(nullptr), right(nullptr) {}
};

class UnbalancedTree {
private:
    Node* root;

    Node* insertRec(Node* node, int value) {
        if (!node) return new Node(value);
        if (value < node->data)
            node->left = insertRec(node->left, value);
        else
            node->right = insertRec(node->right, value);
        return node;
    }

    void inorderRec(Node* node) {
        if (!node) return;
        inorderRec(node->left);
        std::cout << node->data << " ";
        inorderRec(node->right);
    }


    Node* searchRec(Node* node, int key) {
        if (!node || node->data == key) return node;
        if (key < node->data)
            return searchRec(node->left, key);
        return searchRec(node->right, key);
    }

    Node* findMin(Node* node) {
        while (node->left) node = node->left;
        return node;
    }


    Node* deleteRec(Node* node, int key) {
        if (!node) return node;

        if (key < node->data)
            node->left = deleteRec(node->left, key);
        else if (key > node->data)
            node->right = deleteRec(node->right, key);
        else {
            
            if (!node->left && !node->right) {
                delete node;
                return nullptr;
            }
            
            if (!node->left) {
                Node* temp = node->right;
                delete node;
                return temp;
            }
            if (!node->right) {
                Node* temp = node->left;
                delete node;
                return temp;
            }
            
            Node* temp = findMin(node->right);
            node->data = temp->data;
            node->right = deleteRec(node->right, temp->data);
        }
        return node;
    }

public:
    UnbalancedTree() : root(nullptr) {}

    void insert(int value) {
        root = insertRec(root, value);
    }

    void inorder() {
        inorderRec(root);
        std::cout << std::endl;
    }

    bool search(int key) {
        return searchRec(root, key) != nullptr;
    }

    void remove(int key) {
        root = deleteRec(root, key);
    }
};

int main() {
    UnbalancedTree tree;
    tree.insert(10);
    tree.insert(5);
    tree.insert(20);
    tree.insert(15);
    tree.insert(30);

    std::cout << "Inorder traversal: ";
    tree.inorder();

    std::cout << "Searching for 15: " << (tree.search(15) ? "Found" : "Not Found") << std::endl;

    tree.remove(20);
    std::cout << "Inorder traversal after deleting 20: ";
    tree.inorder();

    return 0;
}
