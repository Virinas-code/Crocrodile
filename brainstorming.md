# Brainstorming Crocrodile
## Comment prendre 1 neurone sur 12 ?
- Boucles for et if ?
 -> Peut-être plus long
 -> On se remet a parcourir tous les ~800 éléments
 -> On utilise ds modulos
- Au moment de la génération ?
 -> Plus complexe
- Sous listes ?
 -> Marche pas pour tout

:warning: Ne pas oublier les dernières entrées !
:warning: Promotion - en quoi ?

----------
## Taille et architecture du réseau
### Types de pièces
- **Carré :** Pions blancs
- **Rond :** Cavaliers blancs
- **Triangle :** Fous blancs
- **Triangle en bas :** Tours blanches
- **Diamant :** Dames blanches
- **Bonhomme :** Roi blanc
- **Scie :** Pions noirs
- **Charlemagne :** Cavaliers noirs
- **Pétard :** Fous noirs
- **Sapin moche :** Tours noires
- **Beau sapin :** Dames noires
- **Œuf de Pâques :** Roi noir

On sépare la couche d'entrées en 12 vecteurs de 109 éléments :

 `64 (pièces) + 4 (roques) + 8 (en passant) + 32 (coup à jouer) + 1 (biais)`

### Cases noires / cases blanches
- **Chocolat blanc :** Cases noires pièces blanches
- **Chocolat noir :** Cases noires pièces noires
- **Blanc chocolat :** Cases blanches pièces blanches
- **Noir chocolat :** Cases blanches pièces blanches

Si pièce on met *1*, sinon *0*

Structure :

 `32 (pièce ou pas) + 4 (roques) + 8 (en passant) + 32 (coup à jouer) + 1 (biais)`

### 3/16 de l'échiqiuer
- **District 1 :** Haut gauche
- **District 2 :** Haut droite
- **District 3 :** Bas droite
- **District 4 :** Bas gauche

3 cases en hauteur, 4 cases en longueur dans le coin.

Sur les lignes 6, 7 et 8 : uniquement les pièces noires.

Sur les lignes 1, 2 et 3 : uniquement les pièces blanches.

Structure :

 `72 (pièces) + 2 (roques de notre côté) + 4 (en passant dans notre quartier) + 32 (coup à jouer) + 1 (biais)`

### Petit centre
**Capitole :** Petit centre

Case haut gauche - haut droite - bas droite - bas gauche

Structure :
 `48 (pièces) + 4 (roques) + 8 (en passant) + 32 (coup à jouer) + 1 (biais)`

### Centre étendu
**Banlieue :** Centre étendu

Case haut gauche - haut droite - bas droite - bas gauche

Structure :
 `144 (pièces) + 4 (roques) + 8 (en passant) + 32 (coup à jouer) + 1 (biais)`

### Lignes et colonnes
**Pêche 1-8 :** Lignes 1-8
**Temple 1-8 :** Colonnes 1-8

Structure :
 `96 (pièces) + 4 (roques) + 32 (coup à jouer) + 1 (biais)`
