import numpy as np 

def jouer_arete(x1, y1, x2, y2):
    if x1 == x2:  # arête verticale
        if abs(y1 - y2) == 1 and arêtes_verticales[min(y1, y2)][x1] == 0: # si la longeur est de 1 et que l'arrete est libre
            arêtes_verticales[min(y1, y2)][x1] = 1
            return True
    elif y1 == y2:  # arête horizontale
        if abs(x1 - x2) == 1 and arêtes_horizontales[y1][min(x1, x2)] == 0:
            arêtes_horizontales[y1][min(x1, x2)] = 1
            return True
    return False

def afficher_grille(N):
    # Affichage des numéros de colonne
    print(" ", end="")
    for i in range(N+1):
        print(f"{i:3}", end="")
    print()

    for y in range(N + 1):
        ligne = ""
        # Affichage du numéro de ligne
        print(f"{y:2} ", end="")
        for x in range(N):
            ligne += "+" if angles[y][x] == 0 else "+"
            ligne += "--" if arêtes_horizontales[y][x] == 1 else "  "
        ligne += "+"
        print(ligne)
        if y < N:
            ligne = "   "
            for x in range(N + 1):
                ligne += "|" if arêtes_verticales[y][x] == 1 else " "
                ligne += "  "
            print(ligne)

def init_plateau(arêtes_array, orientatrion, no_limit=False):
    if no_limit:
        return (arêtes_array)
    else: 
        if (orientatrion=="verticale"):
            for k in range (len(arêtes_array)) :
                arêtes_array[k][0]=1
                arêtes_array[k][-1]=1
        elif(orientatrion=="horizontale"):
            for k in range (len(arêtes_array[0])) :
                arêtes_array[0][k]=1
                arêtes_array[-1][k]=1

        else:
            print("error ni verticale ni horizontale")
            return(False)
        return(arêtes_array)

def generate_board(N, no_limit=False):
    angles = np.zeros((N+1, N+1), dtype=int)
    # Grille des arêtes
    arêtes_horizontales =np.zeros((N + 1, N), dtype=int)
    init_plateau(arêtes_horizontales, "horizontale")
    arêtes_verticales = np.zeros((N , N + 1), dtype=int)
    init_plateau(arêtes_verticales,"verticale")

    return(angles,arêtes_horizontales,arêtes_verticales)

def scorer(arêtes_horizontales, arêtes_verticales):
    cases = np.zeros((N, N), dtype=int)
    score = 0
    for i in range(N):
        for j in range(N):
            if (arêtes_horizontales[i][j] == 1 and 
                arêtes_horizontales[i + 1][j] == 1 and 
                arêtes_verticales[i][j] == 1 and 
                arêtes_verticales[i][j + 1] == 1):
                if cases[i][j] == 0:
                    cases[i][j] = 1
                    score += 1
                    print(f"Case complétée en ({i}, {j})")
    return score

def fin_partie(arêtes_horizontales,arêtes_verticales):
    if(np.any(arêtes_horizontales==0) or np.any(arêtes_verticales == 0)):
        return(False)
    else:
        print("Fin de partie")
        afficher_grille(N)
        return(True)

def tour_de_jeu(score_j1, score_j2, arêtes_horizontales, arêtes_verticales, tour, recompense_j1, recompense_j2):
    current_j=1
    tour += 1
    
    while(current_j==1):
        score_local=scorer(arêtes_horizontales, arêtes_verticales)
        print("tour :" , tour)
        print("Score :")
        print("J1 :", score_j1)
        print("J2 :", score_j2)
        print("C'est au tour du joueur :", current_j)
        afficher_grille(N)
        x1=int(input("x1= ? "))
        y1=int(input("y1= ? "))
        direction=input("right (r) or down (d) ? ")
        if direction == "r":
            y2=y1
            x2=x1+1
        elif direction == "d":
            y2=y1+1
            x2=x1
        else :
            y2=-1
            x2=-1
        if (jouer_arete(x1, y1, x2, y2)):
            if ( scorer(arêtes_horizontales, arêtes_verticales) == score_local):
                current_j=2
            else:
                score_j1=(scorer(arêtes_horizontales, arêtes_verticales)- score_local)
                recompense_j1+=10
        else: 
            print("Coup non valable, merci de rejouer un coup valable")
        if fin_partie(arêtes_horizontales,arêtes_verticales):
            break

    while(current_j==2):
        score_local=scorer(arêtes_horizontales, arêtes_verticales)
        print("tour :" , tour)
        print("Score :")
        print("J1 :", score_j1)
        print("J2 :", score_j2)
        print("C'est au tour du joueur :", current_j)
        afficher_grille(N)
        x1=int(input("x1= ? "))
        y1=int(input("y1= ? "))
        direction=input("right (r) or down (d) ? ")
        if direction == "r":
            y2=y1
            x2=x1+1
        elif direction == "d":
            y2=y1+1
            x2=x1
        else :
            y2=-1
            x2=-1
        if (jouer_arete(x1, y1, x2, y2)):
            if ( scorer(arêtes_horizontales, arêtes_verticales) == score_local):
                current_j=1
            else:
                score_j2=(scorer(arêtes_horizontales, arêtes_verticales)- score_local)
                recompense_j2+=10
        else: 
            print("Coup non valable, merci de rejouer un coup valable")
        if fin_partie(arêtes_horizontales,arêtes_verticales):
            break
    return(score_j1,score_j2)
#####################################################

N=5
score_j1=0
score_j2=0
recompense_j1=0
recompense_j2=0
angles,arêtes_horizontales,arêtes_verticales=generate_board(N)
tour=0
while (not fin_partie(arêtes_horizontales,arêtes_verticales)):
    score_j1, score_j2=tour_de_jeu(score_j1, score_j2, arêtes_horizontales, arêtes_verticales,tour, recompense_j1, recompense_j2)
if score_j1>score_j2:
    recompense_j1+=1000
    recompense_j2-=1000
    print("Félicitation joueur J1 tu as gagné ! ")
elif score_j1 == score_j2:
    recompense_j1+=250
    recompense_j2+=250
    print ("Egalité, bien joué à vous deux !")
else: 
    recompense_j2+=1000
    recompense_j1-=250
    print("Félicitation joueur J2 tu as gagné ! ")
