#!/usr/bin/env bash
# Ablation du conditionnement exogene dynamique.
#
# Trois variantes, chacune dans sa propre experience MLflow pour que le
# balayage POD ne les melange pas avec les runs principaux :
#
#   complet       FiLM conditionne par [z_lb || h_t], h_t variant a chaque
#                 pas de l'horizon
#   statique      FiLM conditionne par [z_lb || h_H], fige sur l'horizon.
#                 h_H a deja parcouru les 24 pas exogenes, donc le CONTENU
#                 est identique : seule disparait la resolution temporelle.
#                 C'est le conditionnement des INR existants, et c'est
#                 l'ablation qui isole reellement le mecanisme propose.
#   sans exogene  FiLM conditionne par z_lb seul. Mesure la valeur des
#                 covariables, pas celle du mecanisme -- a garder comme
#                 borne, pas comme demonstration.
#
# On regarde l'ecart a p=0 (la contribution architecturale) ET sous
# masquage : quand le lookback disparait, le modele depend davantage des
# exogenes, donc l'ecart devrait se creuser. Si c'est le cas, la
# contribution architecturale se relie a la these sur la robustesse.
#
# Usage :  bash ablation_conditioning.sh DE

set -e
MARKET=${1:-DE}
SEEDS="0 1 2 3 4"
RUN="uv run runner.py --config-name=inr dataset.name=$MARKET"
POD="uv run partial_observability_degradation.py --config-name=inr dataset.name=$MARKET"

echo "############ conditionnement statique ############"
for s in $SEEDS; do
  $RUN seed=$s model.static_exog=true exp_suffix=static
done

echo "############ sans exogene ############"
for s in $SEEDS; do
  $RUN seed=$s model.use_exog=false exp_suffix=noexog
done

echo "############ balayages ############"
$POD model.static_exog=true exp_suffix=static
$POD model.use_exog=false exp_suffix=noexog

echo
echo "Comparer aux runs principaux (sans suffixe) :"
echo "  uv run python summarize_pod.py --market $MARKET"
echo
echo "L'ecart complet/statique a p=0 mesure la contribution architecturale."
echo "S'il se creuse avec le taux de masquage, le mecanisme dynamique"
echo "compte d'autant plus que le lookback se raréfie."