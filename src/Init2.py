from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt


def compute_next_step(local_grid):
    # Placeholder for the actual computation logic
    for row in range(1, local_grid.shape[0] - 1):
        for col in range(local_grid.shape[1]):
            local_grid[row, col] = (local_grid[row-1, col] + local_grid[row+1, col]) / 2

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
    grid_start = (rank_sl) * (grid_size // nbp_sl) + min(rank_sl, grid_size % nbp_sl)


    local_grid = np.zeros((local_size + 2, grid_size))
    if rank_sl == 0:
        local_grid[1] = np.ones(grid_size)  # Initialize the first row of the first slave to 1 for testing
        local_grid = np.ones((local_size + 2, grid_size))  # Initialize the ghost row above the first row to 0
    if rank_sl == nbp_sl - 1:
        local_grid[-2] = -1*np.ones(grid_size)  # Initialize the last row of the last slave to 1 for testing
    
    
if rank == 1 :
    loc_sizes = globCom.gather(local_grid.shape[0], root=0)


import pygame  as pg

CarryOn = True 
pg.init()
while CarryOn and step < 1000 : 


    if com.name == "Slaves" :
        local_grid = compute_next_step(local_grid)
        new_grid = com.gather(local_grid[1:-1], root =0) 

    if rank == 1:
        n_grid = np.array(new_grid)
        n_grid = n_grid.reshape((grid_size, grid_size))
        globCom.send(n_grid, dest=0, tag=300)  # Send the full grid to the first slave for visualization  
        
    if rank == 0 : 
        grid = globCom.recv(source=1, tag=300)  # Receive the full grid from the first slave
        print("step ", step)
        print(grid)
        step += 1  


    if com.name == "Slaves" :   
        com.send(local_grid[0], dest=(rank_sl - 1)%nbp_sl, tag=100 + rank_sl)  # Send the first row to the previous process
        com.send(local_grid[-1], dest=(rank_sl + 1)%nbp_sl, tag=200 + rank_sl)  # Send the last row to the next process
        local_grid[0] = com.recv(source=(rank_sl - 1)%nbp_sl, tag=200 + (rank_sl - 1)%nbp_sl)  # Receive the last row from the previous process
        local_grid[-1] = com.recv(source=(rank_sl + 1)%nbp_sl, tag=100 + (rank_sl + 1)%nbp_sl)  # Receive the first row from the next process

    com.Barrier()

    step += 1 
    for event in pg.event.get() :
        if event.type == pg.QUIT :
            CarryOn = False
