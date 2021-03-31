# cython: language_level=3

cdef int foo(int i):
    return i*1000

def foowrapper(i):
    return foo(i)

cdef class Bar:
    cdef public double some

    def __cinit__(self, d):
        self.some = d