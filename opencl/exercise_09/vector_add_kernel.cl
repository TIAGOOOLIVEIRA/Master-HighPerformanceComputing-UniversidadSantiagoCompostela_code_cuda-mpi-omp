__kernel void vector_add(__global int *A, __global int *B, __global int *C) {
    
    // Get the index of the current element
    // Do the operation
    int i = get_global_id(0);
    if(i < count)
        C[i] = A[i] + B[i];
}
