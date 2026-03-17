import pygame  as pg
import numpy   as np

class Grille:
    """
    Grille torique décrivant l'automate cellulaire.
    En entrée lors de la création de la grille :
        - dimensions est un tuple contenant le nombre de cellules dans les deux directions (nombre lignes, nombre colonnes)
        - init_pattern est une liste de cellules initialement vivantes sur cette grille (les autres sont considérées comme mortes)
        - color_life est la couleur dans laquelle on affiche une cellule vivante
        - color_dead est la couleur dans laquelle on affiche une cellule morte
    Si aucun pattern n'est donné, on tire au hasard quels sont les cellules vivantes et les cellules mortes
    Exemple :
       grid = Grille( (10,10), init_pattern=[(2,2),(0,2),(4,2),(2,0),(2,4)], color_life=pg.Color("red"), color_dead=pg.Color("black"))
    """
    def __init__(self, rank : int, nbp : int, dim, init_pattern=None, color_life=pg.Color("black"), color_dead=pg.Color("white")):
        import random
        self.dimensions = dim
        self.dimensions_loc = (dim[0]//nbp + (1 if rank < dim[0]%nbp else 0),dim[1])
        self.start_loc = rank * self.dimensions_loc[0] + (dim[0]%nbp if rank >= dim[0]%nbp else 0)

        if init_pattern is not None:
            print("init_pattern", init_pattern)
            self.cells = np.zeros((self.dimensions_loc[0]+2,self.dimensions_loc[1]), dtype=np.uint8)
            indices_i = [v[0]-self.start_loc+1 for v in init_pattern 
                         if v[0] >= self.start_loc and v[0] < self.start_loc+self.dimensions_loc[0]]
            indices_j = [v[1] for v in init_pattern]
            print("indices_i", indices_i," indices_j", indices_j)
            if len(indices_i) > 0:
                self.cells[indices_i,indices_j] = 1
            print(f"rank {rank} : Live cells :", np.where(self.cells == 1))
        else:
            self.cells = np.random.randint(2, size=dim, dtype=np.uint8)
        self.col_life = color_life
        self.col_dead = color_dead
        #print("Live cells :", np.where(self.cells == 1))

    def compute_next_iteration(self):
        """
        Calcule la prochaine génération de cellules en suivant les règles du jeu de la vie
        """
        neighbours_count = sum(np.roll(np.roll(self.cells, i, 0), j, 1) for i in (-1, 0, 1) for j in (-1, 0, 1) if (i != 0 or j != 0))  
        next_cells = (neighbours_count == 3) | (self.cells & (neighbours_count == 2))
        diff_cells = (next_cells != self.cells)
        self.cells = next_cells
        return diff_cells

    def update_ghost_cells(self):
        """
        Met à jour les cellules fantômes
        """
        req1 = newCom.Irecv(self.cells[-1,:], source = (newCom.rank+1)%newCom.size, tag=101)
        req2 = newCom.Irecv(self.cells[0,:], source = (newCom.rank+newCom.size-1)%newCom.size, tag=102)
        newCom.Send(self.cells[-2,:], dest = (newCom.rank+1)%newCom.size, tag=102)
        newCom.Send(self.cells[1,:], dest = (newCom.rank+newCom.size-1)%newCom.size, tag=101)
        req1.Wait()
        req2.Wait()

    def modify(self, diff): 
        """
        Parameters 
        ------------
        diff : TYPE 
            Modifies Indicated Cells.
        Returns 
        ------------
        None
        """
        nx = self.dimensions[1]
        cells_bef = self.cells
        print("Modify diff", diff)
        print("cells.shape :", self.cells)
        for c in diff :
            nr = int(c[0])
            nc = int(c[1]) 
            print("nr :", nr)
            print("nc :", nc)
            print("self.cells[nr,nc]:",self.cells[nr,nc])
            self.cells[nr,nc] = (1 - self.cells[nr,nc])
            print("self.cells[nr,nc] aft :",self.cells[nr,nc])
        cells_after = self.cells 
        print("cells.shape :", cells_bef)
        print("cells_after :", cells_after)
        print("cells.bef == cells.after :", (cells_bef == cells_after))
        return None 


dico_patterns = { # Dimension et pattern dans un tuple
        'blinker' : ((5,5),[(2,1),(2,2),(2,3)]),
        'toad'    : ((6,6),[(2,2),(2,3),(2,4),(3,3),(3,4),(3,5)]),
        "acorn"   : ((100,100), [(51,52),(52,54),(53,51),(53,52),(53,55),(53,56),(53,57)]),
        "beacon"  : ((6,6), [(1,3),(1,4),(2,3),(2,4),(3,1),(3,2),(4,1),(4,2)]),
        "boat" : ((5,5),[(1,1),(1,2),(2,1),(2,3),(3,2)]),
        "glider": ((10,9),[(1,1),(2,2),(2,3),(3,1),(3,2)]),
        "glider_gun": ((200,100),[(51,76),(52,74),(52,76),(53,64),(53,65),(53,72),(53,73),(53,86),(53,87),(54,63),(54,67),(54,72),(54,73),(54,86),(54,87),(55,52),(55,53),(55,62),(55,68),(55,72),(55,73),(56,52),(56,53),(56,62),(56,66),(56,68),(56,69),(56,74),(56,76),(57,62),(57,68),(57,76),(58,63),(58,67),(59,64),(59,65)]),
        "space_ship": ((25,25),[(11,13),(11,14),(12,11),(12,12),(12,14),(12,15),(13,11),(13,12),(13,13),(13,14),(14,12),(14,13)]),
        "die_hard" : ((100,100), [(51,57),(52,51),(52,52),(53,52),(53,56),(53,57),(53,58)]),
        "pulsar": ((17,17),[(2,4),(2,5),(2,6),(7,4),(7,5),(7,6),(9,4),(9,5),(9,6),(14,4),(14,5),(14,6),(2,10),(2,11),(2,12),(7,10),(7,11),(7,12),(9,10),(9,11),(9,12),(14,10),(14,11),(14,12),(4,2),(5,2),(6,2),(4,7),(5,7),(6,7),(4,9),(5,9),(6,9),(4,14),(5,14),(6,14),(10,2),(11,2),(12,2),(10,7),(11,7),(12,7),(10,9),(11,9),(12,9),(10,14),(11,14),(12,14)]),
        "floraison" : ((40,40), [(19,18),(19,19),(19,20),(20,17),(20,19),(20,21),(21,18),(21,19),(21,20)]),
        "block_switch_engine" : ((400,400), [(201,202),(201,203),(202,202),(202,203),(211,203),(212,204),(212,202),(214,204),(214,201),(215,201),(215,202),(216,201)]),
        "u" : ((200,200), [(101,101),(102,102),(103,102),(103,101),(104,103),(105,103),(105,102),(105,101),(105,105),(103,105),(102,105),(101,105),(101,104)]),
        "flat" : ((200,400), [(80,200),(81,200),(82,200),(83,200),(84,200),(85,200),(86,200),(87,200), (89,200),(90,200),(91,200),(92,200),(93,200),(97,200),(98,200),(99,200),(106,200),(107,200),(108,200),(109,200),(110,200),(111,200),(112,200),(114,200),(115,200),(116,200),(117,200),(118,200)])
    }

choice = 'glider'

init_pattern = dico_patterns[choice]
grid = Grille(0, 1, *init_pattern)

diff= np.zeros((4,2))
diff[0,0] = 4
diff[0,1] = 4
diff[1,0] = 5
diff[1,1] = 3
diff[2,0] = 6
diff[2,1] = 3
diff[3,0] = 7
diff[3,1] = 4

grid.modify(diff)
