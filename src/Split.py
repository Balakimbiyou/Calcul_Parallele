from mpi4py import MPI


globCom = MPI.COMM_WORLD.Dup()
rank = globCom.Get_rank()
nbp  = globCom.Get_size()

keys = [0,1,0,1,0,1,0]


master = [0]
slaves = [1 for i in range(nbp - 1)]
splitkey = master + slaves 

print("splitkey :", splitkey)

com = globCom.Split(color = splitkey[rank] , key = rank )

print(" com :", com)


if rank != 0 :
    com.Set_name("Slaves")
    slvs_size = com.Get_size()
    slvs_rank = com.Get_rank()
else :
    com.Set_name("Master")

print("I am ", rank, "and locally", com.name)

if com.name == "Slaves" :
    print("Slave n :", slvs_rank)
    print("Number of slaves :", slvs_size)

    a
