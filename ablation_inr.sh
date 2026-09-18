#!/usr/bin/env bash
# Campagne d'ablation de l'INR, sur donnees completes (rate = 0).
#
# Chaque variante est activee seule, pour isoler son effet. Une seule graine
# par variante : c'est un tri grossier, pas une mesure. Les variantes
# retenues devront etre confirmees sur plusieurs graines avant d'entrer
# dans le papier -- un ecart inferieur a la dispersion inter-graines ne
# prouve rien.
#
# Usage :  bash ablation_inr.sh DE

set -e
MARKET=${1:-DE}
RUN="uv run runner.py --config-name=inr dataset.name=$MARKET"

echo "=== base ==="
$RUN

echo "=== + LayerNorm dans l'INR ==="
$RUN model.inr.layer_norm=true

echo "=== + connexions residuelles ==="
$RUN model.inr.skip=true

echo "=== + LayerNorm et residuelles ==="
$RUN model.inr.layer_norm=true model.inr.skip=true

echo "=== + RevIN (normalisation par fenetre) ==="
$RUN model.revin=true

echo "=== - LayerNorm sur le code de fenetre ==="
$RUN model.norm_deepsets=false

echo "=== - frequences apprenables ==="
$RUN model.inr.pe_learnable=false

echo "=== stride d'entrainement 24 (aligne, comme LEAR/DNN) ==="
$RUN window.stride_train=24

echo
echo "Comparer les MAE dans l'experience icassp_${MARKET}_inr."
echo "Confirmer toute variante retenue sur 5 graines avant de conclure :"
echo "  for s in 0 1 2 3 4; do $RUN <option> seed=\$s; done"