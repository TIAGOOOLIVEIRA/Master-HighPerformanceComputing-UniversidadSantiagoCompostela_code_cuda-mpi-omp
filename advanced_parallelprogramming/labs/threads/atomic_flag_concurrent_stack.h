#include <atomic>
#include <thread>

template<typename T>
struct Node {
  T data;
  Node* next;
  Node(const T& data) : data(data), next(nullptr) {}
};


template<typename T>
class ConcurrentStack {
  
  std::atomic_flag  stack_flag;
  std::atomic_flag  pool_flag;

  Node<T> * volatile head;
  Node<T> * volatile pool;
  
  Node<T> *get_node(const T& data)
  { Node<T>* cur_node;

    if(pool == nullptr) {
      return new Node<T>(data);
    } else {
      while (pool_flag.test_and_set());
    
      if(pool == nullptr) {
        pool_flag.clear();
        return new Node<T>(data);
      } else {
        cur_node = pool;
        pool = cur_node->next;
        pool_flag.clear();
      }
    }

    cur_node->data = data;
    return cur_node;
  }
  
  void eraseList(Node<T> *p)
  {
    while (p != nullptr) {
      Node<T> * const next = p->next;
      delete p;
      p = next;
    }
  }

public:

  ConcurrentStack() :
  stack_flag{ATOMIC_FLAG_INIT},
  pool_flag{ATOMIC_FLAG_INIT},
  head{nullptr},
  pool{nullptr}
  { }
  
  void push(const T& data)
  {
    Node<T>* const new_node = get_node(data);
    
    while (stack_flag.test_and_set());

    new_node->next = head;
    head = new_node;
    
    stack_flag.clear();
  }
  
  void pop(T& data)
  { Node<T>* cur_node;

    do {

      // No need to lock for just reading
      while(head == nullptr) { }
      
      while (stack_flag.test_and_set());
      
      cur_node = head;
      if (cur_node != nullptr) {
        head = cur_node->next;
        stack_flag.clear();
        break;
      }
      
      stack_flag.clear();
    
    } while(1);

    data = cur_node->data;

    while (pool_flag.test_and_set());
    
    cur_node->next = pool;
    pool = cur_node;
    
    pool_flag.clear();
  }

  bool empty() const noexcept { return head == nullptr; }

  ~ConcurrentStack()
  {
    eraseList(head);
    eraseList(pool);
  }

};
