binaries=parallel_phonebook prod_cons_cond prod_cons_atom atomic_concurrent_stack atomic_flag_concurrent_stack mutex_concurrent_stack
CPPFLAGS=-std=c++14 -march=native -O3 -DNDEBUG

ifeq ($(shell uname), Darwin)
# One case also set the MACOSX_DEPLOYMENT_TARGET environment variable
	CPPFLAGS+=-mmacosx-version-min=10.12
endif

all : $(binaries)

parallel_phonebook : parallel_phonebook.cpp
	$(CXX) $(CPPFLAGS) -o $@ $< -lpthread

prod_cons_cond: prod_cons_cond.cpp prod_cons_common.cpp
	$(CXX) $(CPPFLAGS) -o $@ $< -lpthread

prod_cons_atom: prod_cons_atom.cpp prod_cons_common.cpp
	$(CXX) $(CPPFLAGS) -o $@ $< -lpthread

atomic_concurrent_stack : concurrent_stack.cpp atomic_concurrent_stack.h
	$(CXX) $(CPPFLAGS) -DATOMIC_STACK -o $@ $< -lpthread -latomic

atomic_flag_concurrent_stack : concurrent_stack.cpp atomic_concurrent_stack.h
	$(CXX) $(CPPFLAGS) -DATOMIC_FLAG_STACK -o $@ $< -lpthread -latomic

mutex_concurrent_stack : concurrent_stack.cpp mutex_concurrent_stack.h
	$(CXX) $(CPPFLAGS) -o $@ $< -lpthread

clean:
	-@rm -rf $(binaries) *.o *.dSYM
