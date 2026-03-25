"""
Le jeu de la vie
################
Le jeu de la vie est un automate cellulaire inventé par Conway se basant normalement sur une grille infinie
de cellules en deux dimensions. Ces cellules peuvent prendre deux états :
    - un état vivant
    - un état mort
A l'initialisation, certaines cellules sont vivantes, d'autres mortes.
Le principe du jeu est alors d'itérer de telle sorte qu'à chaque itération, une cellule va devoir interagir avec
les huit cellules voisines (gauche, droite, bas, haut et les quatre en diagonales.) L'interaction se fait selon les
règles suivantes pour calculer l'irération suivante :
    - Une cellule vivante avec moins de deux cellules voisines vivantes meurt ( sous-population )
    - Une cellule vivante avec deux ou trois cellules voisines vivantes reste vivante
    - Une cellule vivante avec plus de trois cellules voisines vivantes meurt ( sur-population )
    - Une cellule morte avec exactement trois cellules voisines vivantes devient vivante ( reproduction )

Pour ce projet, on change légèrement les règles en transformant la grille infinie en un tore contenant un
nombre fini de cellules. Les cellules les plus à gauche ont pour voisines les cellules les plus à droite
et inversement, et de même les cellules les plus en haut ont pour voisines les cellules les plus en bas
et inversement.

On itère ensuite pour étudier la façon dont évolue la population des cellules sur la grille.
"""
import pygame  as pg
import numpy   as np
from mpi4py import MPI

globCom = MPI.COMM_WORLD.Dup()
rank = globCom.Get_rank()
nbp  = globCom.Get_size()

newCom = globCom.Split(rank != 0, rank)
print(f"rang global : {rank}, rang local : {newCom.Get_rank()}, nb de processus locaux : {newCom.Get_size()}")

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
        print(f"rank {rank} : dimensions globales : {self.dimensions}")
        if rank %2 == 0 :
            if rank == nbp - 1 :
                block_dim = (dim[0]//nbp + dim[0]%nbp ,dim[1])
            else : 
                block_dim = (dim[0]//nbp , dim[1]//nbp)
        else :
            block_dim = (dim[0]//nbp , dim[1]//nbp )

        # Find local dimensions
        if nbp % 2 == 0 :
            if rank % 2 == 0 :
                
                self.dimensions_loc = (2*block_dim[0], (nbp//2)*block_dim[1])
                if rank + 2 == nbp :
                    self.dimensions_loc = (2*block_dim[0] + dim[0]%nbp, (nbp//2)*block_dim[1]) 
            else : 
                self.dimensions_loc = (2*block_dim[0], (nbp//2)*block_dim[1] + dim[1]%nbp)
                if rank + 1 == nbp :
                    self.dimensions_loc = (2*block_dim[0] + dim[0]%nbp, (nbp//2)*block_dim[1] + dim[1]%nbp) 
        else : 
            if rank %2 == 0:
                if rank == nbp - 1 :
                    self.dimensions_loc = (block_dim[0], dim[1])
                else :
                    self.dimensions_loc = (2*block_dim[0], (nbp//2)*block_dim[1] + block_dim[1]//2 ) 
            else : 
                self.dimensions_loc = (2*block_dim[0], (nbp//2)*block_dim[1] + block_dim[1]//2 + dim[1]%nbp) 


        if nbp % 2 == 0 :
            if rank%2 == 0 :
                self.start_loc_col = 0 
                self.start_loc_row = (rank//2)*2*block_dim[0] 
            else :
                self.start_loc_col = (nbp//2)*block_dim[1] 
                self.start_loc_row = (rank//2)*2*block_dim[0] 

        else : 
            if rank < nbp - 1:
                if rank%2 == 0 :
                    self.start_loc_col = 0 
                    self.start_loc_row = (rank//2)*2*block_dim[0] 
                else :
                    self.start_loc_col = (nbp//2)*block_dim[1] + block_dim[1]//2 
                    self.start_loc_row = (rank//2)*2*block_dim[0] 
            else : 
                self.start_loc_row = 2*(rank//2)*(block_dim[0] - dim[0]%nbp)
                self.start_loc_col = 0
            

        print(f"rank {rank} : dimensions locales : {self.dimensions_loc}, start_loc_row : {self.start_loc_row}, star_loc_col : {self.start_loc_col}")

        if init_pattern is not None:
            #print("init_pattern", init_pattern)
            self.cells = np.zeros((self.dimensions_loc[0]+2,self.dimensions_loc[1]+2), dtype=np.uint8)
            indices_i = [v[0]-self.start_loc_row+1 for v in init_pattern 
                         if v[0] >= self.start_loc_row and v[0] < self.start_loc_row+self.dimensions_loc[0]]
            indices_j = [v[1] - self.start_loc_col +1 for v in init_pattern 
                         if v[1] >= self.start_loc_col and v[1] < self.start_loc_col + self.dimensions_loc[1]]
            #print("indices_i", indices_i," indices_j", indices_j)
            if len(indices_i) and len(indices_j) > 0:
                self.cells[indices_i,indices_j] = 1            
            #print(f"rank {rank} : Live cells :", np.where(self.cells == 1))
        else:
            #print(f"rank {rank} : tirage aléatoire des cellules vivantes et mortes")
            self.cells = np.array(np.random.randint(2, size=dim, dtype=np.uint8))
        self.col_life = color_life
        self.col_dead = color_dead

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

        if (nbp-1) % 2 == 0 :
            ### rows


            down = (newCom.rank+2)%newCom.size
            up = (newCom.rank-2)%newCom.size

            req1 = newCom.Irecv(self.cells[-1,:], source = down, tag=101)
            req2 = newCom.Irecv(self.cells[0,:], source = up, tag=102)
            newCom.Send(self.cells[-2,:], dest = down, tag=102)
            newCom.Send(self.cells[1,:], dest = up, tag=101)
            req1.Wait()
            req2.Wait()

            ### cols 

            col_1_temp = np.ascontiguousarray(self.cells[:,-2])
            col_m2_temp = np.ascontiguousarray(self.cells[:, 1])


            col_m1_temp = np.empty((self.dimensions_loc[0] +2), dtype=np.uint8)
            col_0_temp = np.empty((self.dimensions_loc[0] + 2), dtype=np.uint8)

            partner_col = (newCom.rank + 1) if (newCom.rank %2 ==0) else (newCom.rank -1)

            if newCom.rank %2 == 0 : #left

                req3 = newCom.Isend(col_m2_temp, dest = (partner_col), tag=103)
                newCom.Recv(col_m1_temp, source = (partner_col), tag=104)
                req3.Wait()

                req4 = newCom.Isend(col_1_temp, dest = (partner_col), tag=105)
                newCom.Recv(col_0_temp, source = (partner_col), tag=106)
                req4.Wait()

            else : #right 

                newCom.Recv(col_m1_temp, source = (partner_col), tag=103)
                req3 = newCom.Isend(col_m2_temp, dest = (partner_col), tag=104)
                req3.Wait()

                newCom.Recv(col_0_temp, source = (partner_col), tag=105)
                req4 = newCom.Isend(col_1_temp, dest = (partner_col), tag=106)
                req4.Wait()
            
            self.cells[:,-1]= col_m1_temp
            self.cells[:,0] = col_0_temp

        """
        else :
            #print("nbp-1 :", nbp- 1, " (nbp-1) % 2 :", (nbp-1) % 2 )
            if newCom.rank == newCom.size - 1:


                ### rows
                mid_lig = self.dimensions_loc[1]//2
                #print("mid_lig :", mid_lig)

                req1 = newCom.Irecv(self.cells[-1,:mid_lig], source = 0, tag=101)
                req2 = newCom.Irecv(self.cells[0,:mid_lig], source = (newCom.rank-2), tag=102)
                newCom.Send(self.cells[-2,:mid_lig], dest = 0, tag=102)
                newCom.Send(self.cells[1,:mid_lig], dest = (newCom.rank-2), tag=101)

                req3 = newCom.Irecv(self.cells[-1,mid_lig:], source = 1, tag=103)
                req4 = newCom.Irecv(self.cells[0,mid_lig:], source = (newCom.rank-1), tag=104)
                newCom.Send(self.cells[-2,mid_lig:], dest = 1, tag=104)
                newCom.Send(self.cells[1,mid_lig:], dest = (newCom.rank-1), tag=103)

                ###cols

                self.cells[:,0] = self.cells[:,-2]
                self.cells[:,-1] = self.cells[:,1]

            else :

                ### rows

                if newCom.rank == 0 or newCom.rank == 1: 
                    #print("rank :", newCom.rank, "rank -1 :", newCom.rank -1)

                    req1 = newCom.Isend(self.cells[-2,:], source = (newCom.rank+2)%newCom.size, tag=101)
                    req2 = newCom.Isend(self.cells[1,:], source = (newCom.size-1), tag=102)
                    newCom.Recv(self.cells[-1,:], dest = (newCom.rank+2)%newCom.size, tag=102)
                    newCom.Recv(self.cells[0,:], dest = (newCom.size-1), tag=101)
                
                elif newCom.rank == newCom.size - 3 or newCom.rank == newCom.size - 2:
                    #print("rank :", newCom.rank, "rank -1 :", newCom.rank -1)
                    req1 = newCom.Irecv(self.cells[-1,:], source = (newCom.size-1), tag=101)
                    req2 = newCom.Irecv(self.cells[0,:], source = (newCom.rank-2)%newCom.size, tag=102)
                    newCom.Send(self.cells[-2,:], dest = (newCom.size -1 ), tag=102)
                    newCom.Send(self.cells[1,:], dest = (newCom.rank-2)%newCom.size, tag=101)
                
                else : 
                    #print("rank :", newCom.rank, "rank -1 :", newCom.rank -1)
                    req1 = newCom.Irecv(self.cells[-1,:], source = (newCom.rank+2)%newCom.size, tag=101)
                    req2 = newCom.Irecv(self.cells[0,:], source = (newCom.rank-2)%newCom.size, tag=102)
                    newCom.Send(self.cells[-2,:], dest = (newCom.rank+2)%newCom.size, tag=102)
                    newCom.Send(self.cells[1,:], dest = (newCom.rank-2)%newCom.size, tag=101)

                ### cols 

                col_1_temp = np.ascontiguousarray(self.cells[:,-2])
                col_m2_temp = np.ascontiguousarray(self.cells[:, 1])

                if newCom.rank %2 == 0 :
                    #print("rank :", newCom.rank, "rank + 1 :", newCom.rank + 1)
                    req3 = newCom.Isend(col_m2_temp, dest = (newCom.rank+1), tag=104)
                    req4 = newCom.Isend(col_1_temp, dest = (newCom.rank+1), tag=103)
                    col_m1_temp = np.empty((self.dimensions_loc[0]), dtype=np.uint16)
                    col_0_temp = np.empty((self.dimensions_loc[0]), dtype=np.uint16)
                    newCom.Recv(col_m1_temp, source = (newCom.rank+1), tag=104)
                    newCom.Recv(col_0_temp, source = (newCom.rank+1), tag=103)
                else : 
                    #print("rank :", newCom.rank, "rank -1 :", newCom.rank -1)
                    req3 = newCom.Isend(col_m2_temp, dest = (newCom.rank-1), tag=104)
                    req4 = newCom.Isend(col_1_temp, dest = (newCom.rank-1), tag=103)
                    col_m1_temp = np.empty((self.dimensions_loc[0]), dtype=np.uint16)
                    col_0_temp = np.empty((self.dimensions_loc[0]), dtype=np.uint16)
                    newCom.Recv(col_m1_temp, source = (newCom.rank-1), tag=104)
                    newCom.Recv(col_0_temp, source = (newCom.rank-1), tag=103)

                self.cells[:,-1]= col_m1_temp
                self.cells[:,0] = col_0_temp


        #req1.Wait()
        #req2.Wait()
        #req3.Wait()
        #req4.Wait()
        """
    
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
        for c in diff :
            nr = c //nx 
            nc = c % nx
            self.cells[nr,nc] = (1 - self.cells[nr,nc])
        return None 

class App:
    """
    Cette classe décrit la fenêtre affichant la grille à l'écran
        - geometry est un tuple de deux entiers donnant le nombre de pixels verticaux et horizontaux (dans cet ordre)
        - grid est la grille décrivant l'automate cellulaire (voir plus haut)
    """
    def __init__(self, geometry, grid):
        self.grid = grid
        # Calcul de la taille d'une cellule par rapport à la taille de la fenêtre et de la grille à afficher :
        self.size_x = geometry[1]//grid.dimensions[1]
        self.size_y = geometry[0]//grid.dimensions[0]
        if self.size_x > 4 and self.size_y > 4 :
            self.draw_color=pg.Color('lightgrey')
        else:
            self.draw_color=None
        # Ajustement de la taille de la fenêtre pour bien fitter la dimension de la grille
        self.width = grid.dimensions[1] * self.size_x
        self.height= grid.dimensions[0] * self.size_y
        # Création de la fenêtre à l'aide de tkinter
        self.screen = pg.display.set_mode((self.width,self.height))
        #
        self.canvas_cells = []
        self.colors = np.array([self.grid.col_dead[:-1], self.grid.col_life[:-1]])

    def draw(self):
        surface = pg.surfarray.make_surface(self.colors[self.grid.cells[:,1:-1].T])
        surface = pg.transform.flip(surface, False, True)
        surface = pg.transform.scale(surface, (self.width, self.height))
        self.screen.blit(surface, (0,0))
        if (self.draw_color is not None):
            [pg.draw.line(self.screen, self.draw_color, (0,i*self.size_y), (self.width,i*self.size_y)) for i in range(self.grid.dimensions[0])]
            [pg.draw.line(self.screen, self.draw_color, (j*self.size_x,0), (j*self.size_x,self.height)) for j in range(self.grid.dimensions[1])]
        pg.display.update()


if __name__ == '__main__':
    import time
    import sys

    dico_patterns = { # Dimension et pattern dans un tuple
        'blinker' : ((5,5),[(2,1),(2,2),(2,3)]),
        'toad'    : ((6,6),[(2,2),(2,3),(2,4),(3,3),(3,4),(3,5)]),
        "acorn"   : ((100,100), [(51,52),(52,54),(53,51),(53,52),(53,55),(53,56),(53,57)]),
        "beacon"  : ((6,6), [(1,3),(1,4),(2,3),(2,4),(3,1),(3,2),(4,1),(4,2)]),
        "boat" : ((5,5),[(1,1),(1,2),(2,1),(2,3),(3,2)]),
        "glider": ((100,90),[(1,1),(2,2),(2,3),(3,1),(3,2)]),
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
    if len(sys.argv) > 1 :
        choice = sys.argv[1]
    resx = 800
    resy = 800
    if len(sys.argv) > 3 :
        resx = int(sys.argv[2])
        resy = int(sys.argv[3])
    print(f"Pattern initial choisi : {choice}")
    print(f"resolution ecran : {resx,resy}")
    try:
        init_pattern = dico_patterns[choice]
    except KeyError:
        print("No such pattern. Available ones are:", dico_patterns.keys())
        exit(1)

    
    if rank == 0:
        pg.init()
        grid = Grille(0, 1, *init_pattern)
        appli = App((resx, resy), grid)


        loop = True
        while loop:
            globCom.send(1, dest=1)

            appli.grid.modify(globCom.recv(source=1)) # réception des céllules à modifier et modification de la grille

            t2 = time.time()
            appli.draw()
            t3 = time.time()
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    loop = False
                    pg.quit()
                    globCom.send(-1,dest=1)
            print(f"Temps affichage : {t3-t2:2.2e} secondes", flush=True)

    else:
        grid = Grille(newCom.rank, newCom.size, *init_pattern)
        grid.update_ghost_cells()

        loop = True
        while loop:
            time.sleep(0.1) # A régler ou commenter pour vitesse maxi
            t1 = time.time()
            diff = grid.compute_next_iteration()
            rows, cols = np.where(diff)# indices des cellules à modifier
            mask = (rows >= 1) & ( rows  <= grid.dimensions_loc[0]) &(cols >= 1) & ( cols  <= grid.dimensions_loc[1])
            rows, cols = rows[mask], cols[mask]

            diff_send = []
            if diff[0].shape != (0,):
                diff_send = (grid.start_loc_row+ rows - 1)*grid.dimensions[1] + grid.start_loc_col+ cols - 1
            grid.update_ghost_cells() # mise à jour des cellules fantômes à chaque itération
            t2 = time.time()


            diff_glob = newCom.gather(diff_send, root=0) #regroupement des cellules à modifier sur le processeur 1. Petit gather pour envois sérialisé. Peut-être pas le plus opti.
            if newCom.rank == 0:
                diff_index = [x for xs in diff_glob for x in xs ] # Rassemblement des indices à modifier par proc en une seule liste
                diff_glob = np.unique(diff_index) #unique array pour éviter 
                print("diff_glob2", diff_glob)
                if (globCom.Iprobe(source=0)):
                    a = globCom.recv(source=0) 
                    if a==-1:
                        loop = False
                    else:
                        globCom.send(diff_glob, dest=0)#envois classique bloquant. Peut tenter non bloquant pour tenter d'accélerer le code, attention aux modifs possibles de la liste.
            print(f"Temps calcul prochaine generation : {t2-t1:2.2e} secondes", flush=True)

