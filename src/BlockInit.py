from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt


def compute_next_step(local_grid):
    # Placeholder for the actual computation logic
    for row in range(local_grid.shape[0]):
        for col in range(1, local_grid.shape[1] - 1):
            local_grid[row, col] = (local_grid[row, col-1] + local_grid[row, col+1]) / 2

    return local_grid

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

grid_size = 10
step = 0

if rank == 0 :
    grid = np.zeros((grid_size, grid_size))
    step = 0 

if rank == 1 :
    grid = np.zeros((grid_size, grid_size))

sendcounts = None
displs = None

if com.name == "Slaves" :
    rank_sl = com.Get_rank()
    nbp_sl  = com.Get_size()

    local_size = grid_size // nbp_sl + 1 if rank_sl < grid_size % nbp_sl else grid_size // nbp_sl
    print("I am ", rank, "and I have local size ", local_size)
    grid_start = (rank_sl) * (grid_size // nbp_sl) + min(rank_sl, grid_size % nbp_sl)


    local_grid = np.zeros((grid_size, local_size + 2))

    print("I am ", rank, "and I have local grid of shape ", local_grid.shape)
    if rank_sl == 0:
        #local_grid[1] = np.ones(grid_size)  # Initialize the first row of the first slave to 1 for testing
        local_grid = np.ones((grid_size, local_size + 2))  # Initialize the ghost column to the left of the first column to 0
    if rank_sl == nbp_sl - 1:
        local_grid[:, -2] = -1*np.ones(grid_size)  # Initialize the last column of the last slave to 1 for testing

