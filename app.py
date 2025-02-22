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
        <h1 class="text-4xl font-bold text-center mb-8">⚽ Prédiction de Matchs de Football</h1>
        
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
                        <div id="teamA-stats" class="space-y-2"></div>
                    </div>
                    <div>
                        <h4 class="text-lg font-medium">Équipe B</h4>
                        <div id="teamB-stats" class="space-y-2"></div>
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
        let teamAStats = {};
        let teamBStats = {};

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
        function generateStatsFields() {
            const teamAStatsDiv = document.getElementById('teamA-stats');
            const teamBStatsDiv = document.getElementById('teamB-stats');
            teamAStatsDiv.innerHTML = '';
            teamBStatsDiv.innerHTML = '';

            const stats = [
                { label: 'Buts totaux', key: 'goals' },
                { label: 'Possession moyenne (%)', key: 'possession' },
                { label: 'Expected Goals (xG)', key: 'xg' },
                { label: 'Tirs cadrés par match', key: 'shots' }
            ];

            stats.forEach(stat => {
                teamAStatsDiv.innerHTML += `
                    <div>
                        <label>${stat.label} :</label>
                        <input type="number" class="w-full p-2 border rounded" onchange="updateStats('A', '${stat.key}', this.value)">
                    </div>
                `;
                teamBStatsDiv.innerHTML += `
                    <div>
                        <label>${stat.label} :</label>
                        <input type="number" class="w-full p-2 border rounded" onchange="updateStats('B', '${stat.key}', this.value)">
                    </div>
                `;
            });
        }

        // Mettre à jour les statistiques générales
        function updateStats(team, key, value) {
            if (team === 'A') teamAStats[key] = parseFloat(value);
            else teamBStats[key] = parseFloat(value);
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
        generateStatsFields();
    </script>
</body>
</html>
