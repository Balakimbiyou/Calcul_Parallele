from mpi4py import MPI
import numpy as np

globCom = MPI.COMM_WORLD.Dup()
rank = globCom.Get_rank()
nbp  = globCom.Get_size()


master = [0]
slaves = [1 for i in range(nbp - 1)]
splitkey = master + slaves 

com = globCom.Split(color = splitkey[rank] , key = rank )

if rank != 0 :
    com.Set_name("Slaves")
    slvs_size = com.Get_size()
    slvs_rank = com.Get_rank()

    print("I am ", rank, "and locally", slvs_rank, "of ", slvs_size, "slaves")
else :
    com.Set_name("Master")

print("I am ", rank, "and locally", com.name)

if com.name == "Slaves" :
    print("Slave n :", slvs_rank)
    print("Number of slaves :", slvs_size)

grid_size = 800

grid = None
sendcounts = None
displs = None


if com.name == "Slaves" :

    rank_sl = com.Get_rank()
    nbp_sl  = com.Get_size()

if rank == 1 :
    grid = np.eye(800)
    # Calculate how many rows each process gets
    rows_per_proc = [grid_size // nbp_sl + (1 if i < grid_size % nbp_sl else 0) for i in range(nbp_sl)]
    # Convert to element counts and displacements
    sendcounts = np.array([rows * grid_size for rows in rows_per_proc], dtype=int)
    displs = np.array([sum(sendcounts[:i]) for i in range(nbp_sl)], dtype=int)

if rank == 1:
    sendbuf = [grid, (sendcounts, displs)]
else:
    sendbuf = None

if com.name == "Slaves" :

    local_size = grid_size // nbp_sl + 1 if rank_sl < grid_size % nbp_sl else grid_size // nbp_sl
    print("I am ", rank, "and I have local size ", local_size)
    local_grid = np.zeros((local_size, grid_size))  

    if rank == 1:
        sendbuf = [grid, (sendcounts, displs)]
    else:
        sendbuf = None

    com.Scatterv(sendbuf, local_grid, root=0)
    print("I am ", rank, "and I have local grid of shape ", local_grid.shape)