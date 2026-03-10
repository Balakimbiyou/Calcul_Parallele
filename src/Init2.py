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

if com.name == "Slaves" :
    print("Slave n :", slvs_rank)
    print("Number of slaves :", slvs_size)

grid_size = 10

if rank == 0 :
    grid = np.zeros((grid_size, grid_size))

if rank == 1 :
    grid = np.zeros((grid_size, grid_size))

sendcounts = None
displs = None

if com.name == "Slaves" :
    rank_sl = com.Get_rank()
    nbp_sl  = com.Get_size()

    local_size = grid_size // nbp_sl + 1 if rank_sl < grid_size % nbp_sl else grid_size // nbp_sl
    grid_start = (rank_sl) * (grid_size // nbp_sl) + min(rank_sl, grid_size % nbp_sl)

    #print("I am ", rank, "and I have local size ", local_size)
    #print("grid start ", grid_start)

    local_grid = np.zeros((local_size + 2, grid_size))
    if rank_sl == 0:
        local_grid[1] = np.ones(grid_size)  # Initialize the first row of the first slave to 1 for testing
        local_grid = np.ones((local_size + 2, grid_size))  # Initialize the ghost row above the first row to 0
    if rank_sl == nbp_sl - 1:
        local_grid[-2] = -1*np.ones(grid_size)  # Initialize the last row of the last slave to 1 for testing
    
    #print("I am ", rank, "and I have local grid of shape ", local_grid.shape)



    # Calculate how many rows each process gets
    #rows_per_proc = [grid_size // nbp_sl + (1 if i < grid_size % nbp_sl else 0) for i in range(nbp_sl)]
    # Convert to element counts and displacements
    #sendcounts = np.array([rows * grid_size for rows in rows_per_proc], dtype=int)
    #displs = np.array([sum(sendcounts[:i]) for i in range(nbp_sl)], dtype=int)
    
    
if rank == 1 :
    loc_sizes = globCom.gather(local_grid.shape[0], root=0)


import pygame  as pg

CarryOn = True 
pg.init()
while CarryOn : 


    if com.name == "Slaves" :
        local_grid = compute_next_step(local_grid)
        #print("local_grid :", local_grid)
        print("I am ", rank, "and I have computed my local grid for the next step")
        com.Barrier()  # Synchronize processes after computation
        new_grid = com.gather(local_grid[1:-1], root =0) 
        com.Barrier()  # Synchronize processes after gathering




    if rank == 1:
        n_grid = np.array(new_grid)
        n_grid = n_grid.reshape((grid_size, grid_size))
        print("n_grid :", n_grid)
        print("grid :", np.shape(n_grid))
        #print("Slave ", rank, "sent its local grid to the master")
        com.Barrier()  # Synchronize processes after gathering
        globCom.send(n_grid, dest=0, tag=300)  # Send the full grid to the first slave for visualization  
        #print("Slave ", rank, "sent the full grid to the master for visualization") 
        
    com.Barrier()  # Synchronize processes after sending the grid
    
    if rank == 0 : 
        grid = globCom.recv(source=1, tag=300)  # Receive the full grid from the first slave
        print("Master received full grid of shape ", grid.shape)
        print(grid)
        #plt.imshow(grid, cmap='viridis')
        #plt.colorbar()
        #plt.title("Grid Visualization at Master")
        #plt.pause(0.1)  # Pause to update the plot
        #plt.clf()  # Clear the figure for the next update   
    
    #CarryOn = False  # Exit after receiving the grid for demonstration purposes     

    com.Barrier()  # Synchronize processes before the next iteration  

    if com.name == "Slaves" :   
        com.send(local_grid[0], dest=(rank_sl - 1)%nbp_sl, tag=100 + rank_sl)  # Send the first row to the previous process
        com.send(local_grid[-1], dest=(rank_sl + 1)%nbp_sl, tag=200 + rank_sl)  # Send the last row to the next process
        com.Barrier()  # Synchronize processes after communication  
        local_grid[0] = com.recv(source=(rank_sl - 1)%nbp_sl, tag=200 + (rank_sl - 1)%nbp_sl)  # Receive the last row from the previous process
        local_grid[-1] = com.recv(source=(rank_sl + 1)%nbp_sl, tag=100 + (rank_sl + 1)%nbp_sl)  # Receive the first row from the next process

        print("I am ", rank, "and I am ready for the next step")

    com.Barrier()
    for event in pg.event.get() :
        if event.type == pg.QUIT :
            CarryOn = False
