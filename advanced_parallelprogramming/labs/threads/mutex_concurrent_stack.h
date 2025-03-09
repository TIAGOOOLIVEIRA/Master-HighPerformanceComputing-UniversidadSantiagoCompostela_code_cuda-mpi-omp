#include <mutex>
#include <thread>

template<typename T>
struct Node {
  T data;
  Node* next;
  Node(const T& data) : data(data), next(nullptr) {}
};


template<typename T>
class ConcurrentStack {
  
  std::mutex stack_mutex;
  std::mutex pool_mutex;

  Node<T> * volatile head;
  Node<T> * volatile pool;
  
  Node<T> *get_node(const T& data)
  { Node<T>* cur_node;

    if(pool == nullptr) {
      return new Node<T>(data);
    } else {
      std::lock_guard<std::mutex> guard(pool_mutex);
    
      if(pool == nullptr) {
        return new Node<T>(data);
      } else {
        cur_node = pool;
        pool = cur_node->next;
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
  head{nullptr},
  pool{nullptr}
  {}
  
  void push(const T& data)
  {
    Node<T>* const new_node = get_node(data);
    
    std::lock_guard<std::mutex> guard(stack_mutex);

    new_node->next = head;
    head = new_node;
  }
  
  void pop(T& data)
  { Node<T>* cur_node;

    do {

      // No need to lock for just reading
      while(head == nullptr) { }
      
      std::lock_guard<std::mutex> guard(stack_mutex);
      
      cur_node = head;
      if (cur_node != nullptr) {
        head = cur_node->next;
        break;
      }

    } while(1);

    data = cur_node->data;

    std::lock_guard<std::mutex> guard(pool_mutex);
    
    cur_node->next = pool;
    pool = cur_node;
  }

  bool empty() const noexcept { return head == nullptr; }

  ~ConcurrentStack()
  {
    eraseList(head);
    eraseList(pool);
  }

};
