import pandas as pd
import numpy as np

# --- METTEZ LE NOM DE VOTRE FICHIER CSV ICI ---
csv_filename = "data.csv" 
# ---------------------------------------------

try:
    df = pd.read_csv(csv_filename)

    print("--- 1. Informations Générales et Types de Données ---")
    print("Le DataFrame a {0} lignes et {1} colonnes.".format(df.shape[0], df.shape[1]))
    print("\nTypes de données et valeurs non nulles par colonne :")
    df.info()
    print("\n" + "="*50 + "\n")

    print("--- 2. Recherche de Valeurs Infinies (inf) ---")
    infinite_values = df.isin([np.inf, -np.inf]).sum()
    if infinite_values.sum() == 0:
        print("Bonne nouvelle : Aucune valeur infinie détectée.")
    else:
        print("ATTENTION : Des valeurs infinies ont été détectées !")
        print(infinite_values[infinite_values > 0])
    print("\n" + "="*50 + "\n")

    print("--- 3. Analyse des Valeurs Uniques (pour détecter les colonnes constantes) ---")
    print("Nombre de valeurs uniques par colonne :")
    unique_counts = df.nunique()
    print(unique_counts)
    constant_cols = unique_counts[unique_counts == 1].index.tolist()
    if not constant_cols:
        print("\nBonne nouvelle : Aucune colonne n'est entièrement constante.")
    else:
        print(f"\nATTENTION : La ou les colonnes suivantes sont constantes : {constant_cols}")
        print("Cela peut causer des NaN après la normalisation si elles sont numériques.")

except FileNotFoundError:
    print(f"ERREUR : Le fichier '{csv_filename}' n'a pas été trouvé. Vérifiez le nom du fichier.")
except Exception as e:
    print(f"Une erreur inattendue est survenue : {e}")