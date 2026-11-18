<div align="center">

# FINSG — Financial Narrative & Systemic Graph Intelligence
### Extraction Sémantique & Propagation de Risque par Graphes Temporels

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Architecture](https://img.shields.io/badge/Architecture-ElmasCore%20Charter-blue?style=flat-square)
![Validation](https://img.shields.io/badge/Validation-Walk--Forward%20Zero%20Leakage-10B981?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)

</div>

---

## Problématique Scientifique

Les signaux narratifs extraits des communiqués financiers se propagent-ils le long des interdépendances sectorielles pour anticiper les régimes de volatilité anormale mieux qu'une approche tabulaire isolée ?

## Architecture du Domaine (`src/`)

```mermaid
flowchart LR
    CORP["src/corpus/document_aligner.py"] --> NLP["src/nlp/sentiment_scorer.py"]
    CORP --> GRAPH["src/graph/temporal_graph.py"]
    NLP --> PROP["src/models/graph_propagation.py"]
    GRAPH --> PROP
    PROP --> EVAL["src/evaluation/walk_forward.py"]
```

## Résultats & Datasets (`data/financial_corpus.json`, `data/entity_returns.csv`)

| Indicateur | Valeur observée | Description |
|---|---|---|
| Entités & Documents | `12` émetteurs · `12` dépêches | Corpus et rendements sur 60 jours persistés dans `data/` |
| Arêtes temporelles | `34` arêtes (pré-cutoff) | Graphe de corrélation construit sans fuite future |
| ROC-AUC (Graphe) | `1.0000` | Pouvoir discriminant après propagation 1-hop |
| Macro-F1 | `1.0000` | Équilibre précision/rappel (`artifacts/results.json`) |

## Démarrage Rapide

```bash
pytest
python main.py
```

## Dictionnaire des Variables

| Variable | Type | Description |
|---|---|---|
| `entity_id` | `str` | Symbole ou identifiant d'émetteur |
| `day_index` | `int` | Horodatage discret de publication |
| `polarity` | `float` | Polarité sémantique normalisée `[-1, 1]` |
| `uncertainty_ratio` | `float` | Densité lexicale d'incertitude `[0, 1]` |
| `weight` | `float` | Corrélation historique pré-cutoff |
