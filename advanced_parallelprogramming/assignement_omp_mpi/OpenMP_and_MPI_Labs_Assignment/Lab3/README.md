# Labs 3.- MPI: Topologies and Neighborhood Collectives

In essence, MPI's 2D Cartesian topologies provide a structured way to manage processes for domain decomposition, and nonblocking neighborhood collectives (along with nonblocking point-to-point communication) offer the mechanisms to overlap communication and computation, which is crucial for maximizing performance in stencil-based applications.

MPI's 2D topology architecture and neighborhood collective operations are fundamental for efficient parallelization of tasks like stencil computations, primarily by enabling overlap of communication and computation, akin to pipelining

MPI provides virtual topologies as a convenient way to name processes and define communication patterns, which can also allow MPI to optimize communications. For 2D stencil tasks, Cartesian topologies are particularly relevant.

***Key aspects of MPI's 2D Cartesian topology include***

- Structure: Processes are connected to their neighbors in a virtual grid, which can have cyclic or non-cyclic boundaries. Processes are identified by Cartesian coordinates. Communication between any two processes is still permitted.

- Creation (MPI_Dims_create and MPI_Cart_create): MPI_Dims_create is used to compute a balanced factorization of the total number of processes into the desired dimensions for the process grid. For example, for 12 processes in 2D, it might suggest a 4x3 decomposition. MPI_Cart_create then creates a new communicator with this Cartesian virtual process grid.

- Neighbor Identification (MPI_Cart_shift): The MPI_Cart_shift routine is used to compute the ranks of neighboring processes along a specific dimension. It returns MPI_PROC_NULL if no neighbor exists in that direction. This is particularly useful for finding the left/right or upper/lower neighbors in a 2D grid

- Sub-communicators (MPI_Cart_sub): The MPI_Cart_sub function allows splitting a Cartesian grid into slices. Each slice gets a new communicator, enabling independent collective communications within those slices

- Optimization and Reordering: The reorder argument in MPI_Cart_create can enable MPI to optimize communications by renumbering ranks. Optimized reordering is particularly critical on hierarchical hardware (like clusters of SMP nodes) to minimize inter-node communication, often by making the communicating surfaces of data on each node as "quadratic" or "cubic" as possible. New interfaces, such as MPI_Cart_create_weighted (or MPIX_Dims_weighted_create for portable MPIX routines), have been proposed for MPI 4.1 to allow application and hardware topology awareness, enabling more optimized factorization and reordering to minimize communication time

For stencil tasks, which involve frequent exchanges of "halo" or "ghost cell" data with immediate neighbors, neighborhood collective operations are highly relevant.

module load gcc openmpi/4.0.5_ft3



### Memory of work

- #module load gcc openmpi/4.0.5_ft3

To compile and call original stencil
- #mpicc -o stencil stencil.c printarr_par.c -lm
- #mpirun -np 4 ./stencil 200 10 500

To compile and call 2D Topology stencil - First approach
- #mpicc -o stencil_2d stencil_2d.c printarr_par.c -lm
- #mpirun -np 4 ./stencil_2d 200 10 500

To compile and call 2D Topology stencil - 2D Topology approach
- #mpicc -o stencil_2d2 stencil_2d2.c printarr_par.c -lm
- #mpirun -np 4 ./stencil_2d2 200 10 500

To compile and call 2D Topology stencil - Point-to-point approach
- #mpicc -o stencil_2dp2p stencil_2dp2p.c printarr_par.c -lm
- #mpirun -np 4 ./stencil_2dp2p 200 10 500

_[Work-in-progress]_

### Conclusions