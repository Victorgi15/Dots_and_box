class EnvironnementJeu:
    def __init__(self, taille_grille):
        self.taille_grille = taille_grille
        self.etat_jeu = ...  # Initialiser l'état du jeu (la grille)

    def actions_possibles(self):
        # Retourne les actions possibles à partir de l'état actuel du jeu
        ...

    def etat_suivant(self, action):
        # Met à jour l'état du jeu en fonction de l'action choisie
        ...

    def recompense(self):
        # Calcule la récompense en fonction de l'état actuel du jeu
        ...
