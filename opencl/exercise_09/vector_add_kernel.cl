__kernel void vector_add(__global int *A, __global int *B, __global int *C) {
    //Get the index of the current element
    int i = get_global_id(0);
    //Ensure the index will not point to out of the bound in the arrays
    if(i < count)
        C[i] = A[i] + B[i];
}
