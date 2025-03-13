//File: fileIO.cpp

MPI_Init(&argc, &argv);

int rank, N

MPI_Comm_rank (MPI_COMM_WORLD, &rank);

MPI_Comm_size (MPI_COMM_WORLD, &N);

MPI Status status;

if (argc== 1)

{

}

if (rank == 0)

cerr << "Usage " << argv[0] << " filetoload\n";

exit (1);

MPI_File f;

MP1_File_open (MPI_COMM_WORLD, argv[1], MPI_MODE_RDONLY,

MPI_INFO_NULL, &f);

int blockSize;

MPI_Offset filesize;

MPI_File_get_size (f, &filesize);

// get file size in bytes

21

filesize = sizeof(int);

// convert to number of items

blockSize = filesize / N;

// calculate size of block to read per process 26

int pos rank blockSize;

// initial file position per

process

27

if (rank N-1)

28

blockSize filesize pos;

// get all remaining in last

process

29

31

32

33

34

35

36

unique_ptr<int[]> data make_unique<int[]>(blockSize);

MPI_File_seek(f, pos*sizeof(int), MPI_SEEK_SET);

MPI_File_read (f, data.get(), blockSize, MPI_INT, &status);

MPI_File_close (&f);

sleep (rank);

cout << rank << "read"<< blockSize << " numbers." << endl;

37

for (int i=0;130; 1++)

38

cout << data[i] << " ";

39

cout << " Last one is: " << data[blockSize - 1];
cout << endl;

MPI_finalize();
}
