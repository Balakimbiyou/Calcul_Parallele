import numpy as np
import matplotlib.pyplot as plt


# Nombre de processus

N = np.array([3, 4, 5, 6, 7, 8, 9, 10])
N_spe = np.array([3, 5, 7, 9])      # Utilisé uniquement pour le calcul par bloc

#  Version en ligne
# Vecteur des temps moyen d'affichage par nombre de processus
aff_line = np.array([])

# Vecteur des temps moyen de calcul par nombre de processus
calc_line_3 = np.array([0.0005706191062927246, 0.000654184341430664])

calc_line_4 = np.array([0.0018277091979980468, 0.001743342399597168, 0.0016448097229003907])

calc_line_5 = np.array([0.0019526300430297852, 0.0019397668838500976, 0.0014171924591064454,
                        0.0016147422790527344])

calc_line_6 = np.array([0.0018341989517211914, 0.002018467903137207, 0.002126013278961182,
                        0.001656087875366211, 0.0017449464797973633])

calc_line_7 = np.array([0.0008315587043762207, 0.0005895042419433594, 0.0007729358673095703,
                        0.0007844457626342774, 0.000724344253540039, 0.000671947956085205])

calc_line_8 = np.array([0.0006708731651306152, 0.00066487455368042, 0.0007198162078857422,
                        0.0006380763053894043, 0.0006719627380371094, 0.000508845329284668,
                        0.0005187196731567383])

calc_line_9 = np.array([0.002120358943939209, 0.0023005876541137694, 0.0017362823486328125,
                        0.0022081031799316407, 0.0015951476097106933, 0.0022131218910217284,
                        0.0012430863380432128, 0.0012119112014770507])

calc_line_10 = np.array([0.001673346996307373, 0.00245683479309082, 0.0025193476676940918,
                         0.0023961405754089355, 0.00128718900680542, 0.0017235946655273437,
                         0.0020480208396911623, 0.001992304801940918, 0.0011271200180053712])


calc_line = np.array([np.mean(calc_line_3), np.mean(calc_line_4), np.mean(calc_line_5), np.mean(calc_line_6),
                     np.mean(calc_line_7), np.mean(calc_line_8), np.mean(calc_line_9), np.mean(calc_line_10),])

# Version par Colonne
# Vecteur des temps moyen d'affichage par nombre de processus
aff_col = np.array([])

# Vecteur des temps moyen de calcul par nombre de processus
calc_col_3 = np.array([0.0007009406089782715, 0.0006393694877624512])

calc_col_4 = np.array([0.0019648547172546387, 0.0018105077743530273, 0.0017439804077148437])

calc_col_5 = np.array([0.0019561691284179688, 0.0018973941802978515, 0.001414748191833496,
                       0.0015771946907043458])

calc_col_6 = np.array([0.0019390697479248046, 0.0017518138885498048, 0.0018339214324951173,
                       0.0017749881744384766, 0.001579312801361084])

calc_col_7 = np.array([0.0021622748374938963, 0.0017681040763854981, 0.002164109230041504,
                       0.002112509250640869, 0.0015704302787780762, 0.0014027018547058106])

calc_col_8 = np.array([0.001640575885772705, 0.002050774097442627, 0.0021173539161682127,
                       0.0020442371368408203, 0.0013304533958435059, 0.0021311111450195313,
                       0.0013398299217224122])

calc_col_9 = np.array([0.0021936678886413574, 0.0019779105186462404, 0.0015925540924072266,
                       0.00162324857711792, 0.002296884059906006, 0.0013948278427124024,
                       0.002081193447113037, 0.0012863574028015136])

calc_col_10 = np.array([0.0025410375595092773, 0.0018161396980285645, 0.0023039817810058595,
                        0.002478242874145508, 0.0025780191421508787, 0.0014770002365112305,
                        0.002414046764373779, 0.0019425139427185058, 0.0011912674903869628])

calc_col = np.array([np.mean(calc_col_3), np.mean(calc_col_4), np.mean(calc_col_5), np.mean(calc_col_6),
                     np.mean(calc_col_7), np.mean(calc_col_8), np.mean(calc_col_9), np.mean(calc_col_10),])
# version par Bloc
# Vecteur des temps moyen d'affichage par nombre de processus
aff_bloc_3 = np.array([0.000843865394592285, 0.0007598881721496583])

# Vecteur des temps moyen de calcul par nombre de processus
calc_bloc_3 = np.array([0.000843865394592285, 0.0007598881721496583])

calc_bloc_5 = np.array([0.0012488322257995605, 0.0011774749755859375, 0.0011499834060668946,
                        0.001099830150604248])

calc_bloc_7 = np.array([0.0013224139213562011, 0.001226283073425293, 0.001208136558532715,
                        0.0011980328559875488, 0.0011944303512573242, 0.0010684452056884766])

calc_bloc_9 = np.array([0.001292527675628662, 0.0011411194801330567, 0.001217827796936035,
                        0.0012197141647338867, 0.0012264204025268554, 0.0013394341468811034,
                        0.0012590880393981933, 0.0009693074226379395])

calc_bloc = np.array([np.mean(calc_bloc_3), np.mean(calc_bloc_5), np.mean(calc_bloc_7), np.mean(calc_bloc_9),])

######################################## Les Plots ###########################################


plt.plot(N_spe,calc_bloc, 'o-r', label='cal_bloc')
plt.plot(N,calc_line, 'o-b', label='cal_line')
plt.plot(N, calc_col, 'o-g', label='calc_col')
plt.xlabel('Nombre de processus')
plt.ylabel('Temps')
plt.legend()
plt.title('Comparaison des des templs de calcul moyen de chaque type de '
          'parallélisation en fonction du nombre de processus')
plt.show()
