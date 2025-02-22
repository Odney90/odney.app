<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prédiction de Matchs de Football</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body class="bg-gray-100">
    <div class="container mx-auto p-8">
        <h1 class="text-4xl font-bold text-center mb-8"> Prédiction de Matchs de Football</h1>
        
        <!-- Onglets pour la navigation -->
        <div class="flex justify-center mb-8">
            <button onclick="showTab('form')" class="tab-button bg-blue-500 text-white px-4 py-2 rounded-lg mr-4">Formulaire</button>
            <button onclick="showTab('prediction')" class="tab-button bg-green-500 text-white px-4 py-2 rounded-lg">Prédiction</button>
        </div>

        <!-- Section Formulaire -->
        <div id="form" class="tab-content bg-white p-6 rounded-lg shadow-md">
            <h2 class="text-2xl font-bold mb-4">Statistiques des Équipes</h2>
            
            <!-- Forme récente -->
            <div class="mb-8">
                <h3 class="text-xl font-semibold mb-2">Forme récente (5 derniers matchs)</h3>
                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <h4 class="text-lg font-medium">Équipe A</h4>
                        <div id="teamA-form" class="space-y-2"></div>
                    </div>
                    <div>
                        <h4 class="text-lg font-medium">Équipe B</h4>
                        <div id="teamB-form" class="space-y-2"></div>
                    </div>
                </div>
            </div>

            <!-- Statistiques générales -->
            <div class="mb-8">
                <h3 class="text-xl font-semibold mb-2">Statistiques générales</h3>
                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <h4 class="text-lg font-medium">Équipe A</h4>
                        <div id="teamA-general" class="space-y-2"></div>
                    </div>
                    <div>
                        <h4 class="text-lg font-medium">Équipe B</h4>
                        <div id="teamB-general" class="space-y-2"></div>
                    </div>
                </div>
            </div>

            <-- Statistiques d'attaque -->
            
            <div class="mb-8">
                <h3 class="text-xl font-semibold mb-2">Statistiques d'attaque</h3>
                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <h4 class="text-lg font-medium">Équipe A</h4>
                        <div id="teamA-attack" class="space-y-2"></div>
                    </div>
                    <div>
                        <h4 class="text-lg font-medium">Équipe B</h4>
                        <div id="teamB-attack" class="space-y-2"></div>
                    </div>
                </div>
            </div>

            <!-- Statistiques de défense -->
            <div class="mb-8">
                <h3 class="text-xl font-semibold mb-2">Statistiques de défense</h3>
                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <h4 class="text-lg font-medium">Équipe A</h4>
                        <div id="teamA-defense" class="space-y-2"></div>
                    </div>
                    <div>
                        <h4 class="text-lg font-medium">Équipe B</h4>
                        <div id="teamB-defense" class="space-y-2"></div>
                    </div>
                </div>
            </div>

            <!-- Bouton pour entraîner le modèle -->
            <div class="text-center">
                <button onclick="trainModel()" class="bg-blue-500 text-white px-6 py-2 rounded-lg">Entraîner le Modèle</button>
            </div>
        </div>

        <!-- Section Prédiction -->
        <div id="prediction" class="tab-content bg-white p-6 rounded-lg shadow-md hidden">
            <h2 class="text-2xl font-bold mb-4">Résultats de la Prédiction</h2>
            
            <!-- Graphique des scores de forme -->
            <div class="mb-8">
                <h3 class="text-xl font-semibold mb-2">Scores de Forme</h3>
                <div id="form-chart" class="w-full h-64"></div>
            </div>

            <!-- Résultats de la prédiction -->
            <div class="mb-8">
                <h3 class="text-xl font-semibold mb-2">Prédiction</h3>
                <div id="prediction-result" class="text-center text-2xl font-bold"></div>
            </div>

            <!-- Bouton pour prédire -->
            <div class="text-center">
                <button onclick="predictMatch()" class="bg-green-500 text-white px-6 py-2 rounded-lg">Prédire le Résultat</button>
            </div>
        </div>
    </div>

    <script>
        // Variables globales
        let teamAForm = [];
        let teamBForm = [];
        let teamAGeneral = {};
        let teamBGeneral = {};
        let teamAAttack = {};
        let teamBAttack = {};
        let teamADefense = {};
        let teamBDefense = {};

        // Afficher les onglets
        function showTab(tabName) {
            document.querySelectorAll('.tab-content').forEach(tab => tab.classList.add('hidden'));
            document.getElementById(tabName).classList.remove('hidden');
        }

        // Générer les sliders pour la forme récente
        function generateFormSliders() {
            const teamAFormDiv = document.getElementById('teamA-form');
            const teamBFormDiv = document.getElementById('teamB-form');
            teamAFormDiv.innerHTML = '';
            teamBFormDiv.innerHTML = '';

            for (let i = 1; i <= 5; i++) {
                teamAFormDiv.innerHTML += `
                    <div>
                        <label>Match ${i} :</label>
                        <input type="range" min="0" max="2" step="1" value="1" class="w-full" onchange="updateForm('A', ${i - 1}, this.value)">
                        <span>Victoire | Défaite | Match nul</span>
                    </div>
                `;
                teamBFormDiv.innerHTML += `
                    <div>
                        <label>Match ${i} :</label>
                        <input type="range" min="0" max="2" step="1" value="1" class="w-full" onchange="updateForm('B', ${i - 1}, this.value)">
                        <span>Victoire | Défaite | Match nul</span>
                    </div>
                `;
            }
        }

        // Mettre à jour la forme récente
        function updateForm(team, index, value) {
            if (team === 'A') teamAForm[index] = parseFloat(value);
            else teamBForm[index] = parseFloat(value);
        }

        // Générer les champs pour les statistiques générales
        function generateGeneralFields() {
            const teamAGeneralDiv = document.getElementById('teamA-general');
            const teamBGeneralDiv = document.getElementById('teamB-general');
            teamAGeneralDiv.innerHTML = '';
            teamBGeneralDiv.innerHTML = '';

            const stats = [
                { label: 'Possession moyenne (%)', key: 'possession' },
                { label: 'Passes réussies par match', key: 'passes' },
                { label: 'Tacles réussis par match', key: 'tacles' },
                { label: 'Fautes par match', key: 'fautes' },
                { label: 'Cartons jaunes', key: 'yellow_cards' },
                { label: 'Cartons rouges', key: 'red_cards' }
            ];

            stats.forEach(stat => {
                teamAGeneralDiv.innerHTML += `
                    <div>
                        <label>${stat.label} :</label>
                        <input type="number" class="w-full p-2 border rounded" onchange="updateStats('general', 'A', '${stat.key}', this.value)">
                    </div>
                `;
                teamBGeneralDiv.innerHTML += `
                    <div>
                        <label>${stat.label} :</label>
                        <input type="number" class="w-full p-2 border rounded" onchange="updateStats('general', 'B', '${stat.key}', this.value)">
                    </div>
                `;
            });
        }

        // Générer les champs pour les statistiques d'attaque
        function generateAttackFields() {
            const teamAAttackDiv = document.getElementById('teamA-attack');
            const teamBAttackDiv = document.getElementById('teamB-attack');
            teamAAttackDiv.innerHTML = '';
            teamBAttackDiv.innerHTML = '';

            const stats = [
                { label: 'Buts totaux', key: 'goals' },
                { label: 'Expected Goals (xG)', key: 'xg' },
                { label: 'Tirs cadrés par match', key: 'shots' },
                { label: 'Corners par match', key: 'corners' },
                { label: 'Touches dans la surface adverse', key: 'touches' },
                { label: 'Pénalités obtenues', key: 'penalties' }
            ];

            stats.forEach(stat => {
                teamAAttackDiv.innerHTML += `
                    <div>
                        <label>${stat.label} :</label>
                        <input type="number" class="w-full p-2 border rounded" onchange="updateStats('attack', 'A', '${stat.key}', this.value)">
                    </div>
                `;
                teamBAttackDiv.innerHTML += `
                    <div>
                        <label>${stat.label} :</label>
                        <input type="number" class="w-full p-2 border rounded" onchange="updateStats('attack', 'B', '${stat.key}', this.value)">
                    </div>
                `;
            });
        }

        // Générer les champs pour les statistiques de défense
        function generateDefenseFields() {
            const teamADefenseDiv = document.getElementById('teamA-defense');
            const teamBDefenseDiv = document.getElementById('teamB-defense');
            teamADefenseDiv.innerHTML = '';
            teamBDefenseDiv.innerHTML = '';

            const stats = [
                { label: 'Expected Goals concédés (xG)', key: 'xg_conceded' },
                { label: 'Interceptions par match', key: 'interceptions' },
                { label: 'Dégagements par match', key: 'clearances' },
                { label: 'Arrêts par match', key: 'saves' },
                { label: 'Buts concédés par match', key: 'goals_conceded' },
                { label: 'Aucun but encaissé (oui=1, non=0)', key: 'clean_sheet' }
            ];

            stats.forEach(stat => {
                teamADefenseDiv.innerHTML += `
                    <div>
                        <label>${stat.label} :</label>
                        <input type="number" class="w-full p-2 border rounded" onchange="updateStats('defense', 'A', '${stat.key}', this.value)">
                    </div>
                `;
                teamBDefenseDiv.innerHTML += `
                    <div>
                        <label>${stat.label} :</label>
                        <input type="number" class="w-full p-2 border rounded" onchange="updateStats('defense', 'B', '${stat.key}', this.value)">
                    </div>
                `;
            });
        }

        // Mettre à jour les statistiques
        function updateStats(category, team, key, value) {
            if (team === 'A') {
                if (category === 'general') teamAGeneral[key] = parseFloat(value);
                else if (category === 'attack') teamAAttack[key] = parseFloat(value);
                else if (category === 'defense') teamADefense[key] = parseFloat(value);
            } else {
                if (category === 'general') teamBGeneral[key] = parseFloat(value);
                else if (category === 'attack') teamBAttack[key] = parseFloat(value);
                else if (category === 'defense') teamBDefense[key] = parseFloat(value);
            }
        }

        // Entraîner le modèle
        function trainModel() {
            // Simuler l'entraînement du modèle
            alert("Modèle entraîné avec succès !");
        }

        // Prédire le résultat du match
        function predictMatch() {
            // Calculer les scores de forme
            const scoreA = teamAForm.reduce((a, b) => a + b, 0) / teamAForm.length;
            const scoreB = teamBForm.reduce((a, b) => a + b, 0) / teamBForm.length;

            // Afficher les scores de forme
            Plotly.newPlot('form-chart', [
                { x: ['Équipe A', 'Équipe B'], y: [scoreA, scoreB], type: 'bar' }
            ]);

            // Simuler une prédiction
            const prediction = scoreA > scoreB ? "Victoire de l'Équipe A" : "Victoire de l'Équipe B";
            document.getElementById('prediction-result').textContent = prediction;
        }

        // Initialisation
        generateFormSliders();
        generateGeneralFields();
        generateAttackFields();
        generateDefenseFields();
    </script>
</body>
</html>
