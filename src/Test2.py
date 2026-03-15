from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

import numpy as np

a_loc = np.zeros((6,3), dtype=np.int32, order= "C")
a_loc[1,0] = 5
print("ndim a_loc", np.ndim(a_loc))

print("rank", rank, "a_loc : Live cells :", a_loc)

grid_glob = None
if rank == 0:
   grid_glob = np.zeros((6,size*3), dtype=np.int32)
   #grid_glob = np.zeros((size,6,3), dtype=np.int32)
   print("ndim grid_glob", np.ndim(grid_glob))
   print("np.shape(grid_glob)", np.shape(grid_glob))

for i in range(len(np.shape(a_loc))):
    print(f"rank {rank} : a_loc.shape[{i}] = {np.shape(a_loc)[i]}")

sendcounts = np.array(comm.gather(a_loc.size, root=0))

comm.Gatherv(a_loc, [grid_glob, sendcounts], root=0)
if rank == 0:
    #print("np.shape(grid_glob)", np.shape(grid_glob))
    #a,b,c = np.shape(grid_glob)
    #for i in range(len(np.shape(grid_glob))):
    #  print(f"rank {rank} : a_loc.shape[{i}] = {np.shape(grid_glob)[i]}")
    print("grid_glob : Live cells :", grid_glob)