#include <atomic>
#include <cstdint>

template <typename T>
struct tagged_ptr {

  T * ptr ;
  uintptr_t tag ;

  tagged_ptr(): ptr(nullptr), tag(0) {}
  tagged_ptr( T * p ): ptr(p), tag(0) {}
  tagged_ptr( T * p, uintptr_t n ): ptr(p), tag(n) {}
  T * operator->() const { return ptr; }
};


template <typename T>
using atomic_tagged_ptr = std::atomic<tagged_ptr<T>>;


template<typename T>
struct Node {

  tagged_ptr<Node<T>> next;
  T data;

  Node(const T& data) : data(data), next{nullptr, 0} {}
};


template<typename T>
class ConcurrentStack {
  
  atomic_tagged_ptr<Node<T>> head;
  atomic_tagged_ptr<Node<T>> pool;
  
  Node<T> *get_node(const T& data)
  { Node<T>* cur_node;

    tagged_ptr<Node<T>> pool_copy = pool.load();
    do {
      cur_node = pool_copy.ptr;
      if (cur_node == nullptr) {
        return new Node<T>(data);
      }
    } while(!pool.compare_exchange_weak(pool_copy, {cur_node->next.ptr, pool_copy.tag + 1}));

    cur_node->data = data;
    return cur_node;
  }
  
  void eraseList(tagged_ptr<Node<T>> p)
  {
    while (p.ptr != nullptr) {
      const tagged_ptr<Node<T>> next = p.ptr->next;
      delete p.ptr;
      p = next;
    }
  }

  void push(Node<T>* const new_node, atomic_tagged_ptr<Node<T>>& head)
  {
    new_node->next = head.load();
    
    while(!head.compare_exchange_weak(new_node->next, {new_node, new_node->next.tag + 1})) {}
  }

public:

  ConcurrentStack() :
  head{nullptr},
  pool{nullptr}
  {
    // std::cout << "atomic_tagged_ptr<Node<T>> is_lock_free=" << head.is_lock_free() << '\n';
  }
  
  void push(const T& data)
  {
    push(get_node(data), head);
  }
  
  void pop(T& data)
  { Node<T>* cur_node;
    tagged_ptr<Node<T>> head_copy;

    do {
      do {
        head_copy = head.load();
        cur_node = head_copy.ptr;
      }  while (cur_node == nullptr);
    } while(!head.compare_exchange_weak(head_copy, {cur_node->next.ptr, head_copy.tag + 1}));
    
    data = cur_node->data;
    
    push(cur_node, pool);
  }

  bool empty() const noexcept { return head.load().ptr == nullptr; }

  ~ConcurrentStack()
  {
    eraseList(head.load());
    eraseList(pool.load());
  }

};
