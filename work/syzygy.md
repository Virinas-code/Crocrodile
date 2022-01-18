# Crocrodile
## Work: Syzygy
### Tasks
1. Get a dictionnary with WDLs using a loop
2. Fill this dictionnary with a dictionnary containing DTZ: Move informations.
3. Find the better WDL.
4. Winning: Lowest DTZ / Draw: Maximum DTZ / Losing: Maximum DTZ
### Algorithm
```
{2: {} ; 1: {} ; 0: {} ; -1: {} ; -2: {}} -> WDL
# Remplissage
Pour chaque coup dans échiquier:
  Echiquier + coup -> test_board
  coup -> WDL[WDL Syzygy (test_board)][DTZ Syzygy (test_board)]
# /!\ On est après un coup, alors on prend le WDL MINIMUM !!!!
Minimum (Clés (WDL)) -> best_wdl
Si best_wdl < 0:  # On est gagnant

