import os
import io
import itertools
import random
import traceback
import optuna
import pandas as pd
from flask import Flask, request, render_template, flash, redirect, url_for, Response, session
from medgan import MedGAN
from sdmetrics.reports.single_table import QualityReport

# --- CONFIGURATION ---
app = Flask(__name__)
app.config['SECRET_KEY'] = '123456789'

# --- FONCTIONS DE PARSING ---
def parse_params(form_data):
    """Parse le formulaire en un dictionnaire de listes de valeurs (l'espace de recherche)."""
    param_grid = {}
    
    # Parser les valeurs simples (int, float)
    for key in ['epochs', 'batch_size', 'noise_dim']:
        param_grid[key] = [int(v.strip()) for v in form_data.get(key, '').split(',') if v.strip()]
    for key in ['learning_rate', 'l2scale']:
        param_grid[key] = [float(v.strip()) for v in form_data.get(key, '').split(',') if v.strip()]
        
    # Parser les tuples pour les dimensions des couches
    for key in ['generator_dims', 'discriminator_dims']:
        values_str = form_data.get(key, '')
        param_grid[key] = [tuple(map(int, arch.strip().split(','))) for arch in values_str.split(';') if arch.strip()]
        
    # S'assurer qu'aucune liste n'est vide
    for key, value in param_grid.items():
        if not value:
            raise ValueError(f"Le champ pour l'hyperparamètre '{key}' ne peut pas être vide.")
            
    return param_grid

# --- OBJECTIF POUR L'OPTIMISATION ---
# Cette fonction sera appelée par Optuna, Grid Search, etc.
def objective(params, real_data):
    """Entraîne MedGAN avec des paramètres donnés et retourne un score de qualité."""
    try:
        # 1. Instancier et entraîner le modèle
        model = MedGAN(**params)
        model.fit(real_data.copy())
        
        # 2. Générer des données synthétiques
        synthetic_data = model.generate(n_samples=len(real_data))
        
        # 3. Évaluer la qualité
        report = QualityReport()
        report.generate(real_data, synthetic_data)
        score = report.get_score()
        
        # Optuna gère mal les None, on retourne 0.0 en cas d'échec du score
        return score if pd.notna(score) else 0.0
    
    except Exception as e:
        print(f"Erreur durant l'évaluation avec les paramètres {params}: {e}")
        return 0.0 # Un score faible si l'essai échoue

# --- LOGIQUE DE RECHERCHE ---
def find_best_hyperparams(search_method, param_grid, real_data, n_trials):
    """Lance la recherche d'hyperparamètres et retourne les meilleurs."""
    
    # --- Recherche Bayésienne (Optuna) ---
    if search_method == 'bayesian':
        def optuna_objective(trial):
            params = {}
            for name, values in param_grid.items():
                params[name] = trial.suggest_categorical(name, values)
            return objective(params, real_data)
        
        study = optuna.create_study(direction="maximize")
        study.optimize(optuna_objective, n_trials=n_trials)
        return study.best_params, study.best_value

    # --- Recherche Aléatoire ---
    elif search_method == 'random':
        best_score = -1
        best_params = None
        for i in range(n_trials):
            print(f"Random Search - Essai {i+1}/{n_trials}")
            params = {key: random.choice(values) for key, values in param_grid.items()}
            score = objective(params, real_data)
            if score > best_score:
                best_score = score
                best_params = params
        return best_params, best_score

    # --- Recherche par Grille ---
    elif search_method == 'grid':
        keys, values = zip(*param_grid.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        print(f"Grid Search - Test de {len(param_combinations)} combinaisons.")
        best_score = -1
        best_params = None
        for i, params in enumerate(param_combinations):
            print(f"Grid Search - Essai {i+1}/{len(param_combinations)}")
            score = objective(params, real_data)
            if score > best_score:
                best_score = score
                best_params = params
        return best_params, best_score

    raise ValueError(f"Méthode de recherche non reconnue: {search_method}")

# --- ROUTE PRINCIPALE ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'datafile' not in request.files or request.files['datafile'].filename == '':
            flash("Veuillez sélectionner un fichier.", "danger")
            return redirect(request.url)
        
        file = request.files['datafile']
        try:
            # 1. Charger les données et les paramètres du formulaire
            real_data = pd.read_csv(io.StringIO(file.stream.read().decode('UTF-8')))
            form_data = request.form
            param_grid = parse_params(form_data)
            search_method = form_data.get('search_method', 'bayesian')
            n_trials = int(form_data.get('n_trials', 15))

            # 2. Lancer la recherche des meilleurs hyperparamètres
            best_params, best_score = find_best_hyperparams(search_method, param_grid, real_data, n_trials)
            
            if best_params is None:
                raise RuntimeError("L'optimisation n'a trouvé aucune configuration valide.")

            # 3. Ré-entraîner le modèle final avec les meilleurs paramètres
            print("\nRé-entraînement du modèle final avec les meilleurs paramètres...")
            final_model = MedGAN(**best_params, verbose=True)
            final_model.fit(real_data.copy())
            synthetic_data = final_model.generate(n_samples=len(real_data))

            # 4. Préparer les résultats pour l'affichage
            session['results'] = {
                'best_params': best_params,
                'best_score': best_score,
                'search_method': search_method,
                'n_trials': n_trials,
                'table_html': synthetic_data.head().to_html(classes='table table-sm table-striped', index=False),
                'csv_data': synthetic_data.to_csv(index=False)
            }
            return redirect(url_for('result'))

        except Exception as e:
            traceback.print_exc()
            flash(f"Une erreur est survenue : {e}", "danger")
            return redirect(request.url)

    return render_template('index.html')

@app.route('/result')
def result():
    results = session.get('results')
    if not results:
        return redirect(url_for('index'))
    return render_template('result.html', **results)

@app.route('/download')
def download():
    results = session.get('results')
    if not results or 'csv_data' not in results:
        return redirect(url_for('index'))
    return Response(
        results['csv_data'],
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=optimized_synthetic_data.csv"}
    )

if __name__ == '__main__':
    app.run(debug=True)