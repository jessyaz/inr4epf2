#!/usr/bin/env bash
# Campagne complete : estimation, ablation, balayage.
#
# Cinq bras :
#   lear                 reference du benchmark, estimation en forme close
#   dnn                  seconde reference du benchmark
#   imputed_transformer  attention sur serie imputee
#   masked_transformer   MEME architecture, positions absentes exclues de
#                        l'attention. La paire isole le mode d'ingestion a
#                        architecture constante : meme nombre de
#                        parametres, memes couches, meme entrainement.
#   inr                  le modele propose
#
# Chaque modele est aussi estime SANS lookback (use_lookback=false). Ce
# modele ablate est la borne de rho : il a realloue sa capacite sur les
# seules exogenes, donc il est correctement specifie pour la tache sans
# historique. Le point a rate = 1.0, lui, est un modele entraine AVEC
# lookback puis brutalement prive de tout : il opere hors de son domaine et
# peut diverger, ce qui en ferait une borne trompeuse.
#
# L'imputation est fixee a `linear`. Le choix vient d'une comparaison
# prealable sur DE : en regime ponctuel l'interpolation lineaire degrade
# nettement moins que la saisonniere (LEAR +25 % contre +41 %), la
# saisonniere ne reprenant l'avantage qu'en regime par blocs.
#
# LEAR est deterministe : un seul run par marche. Les autres en ont cinq.
#
# Usage :  bash run_campaign.sh          # tout
#          bash run_campaign.sh train    # estimation seule
#          bash run_campaign.sh pod      # balayages seuls
#          bash run_campaign.sh ablation # ablations du conditionnement

set -u

MARKETS="NP PJM BE FR DE"
SEEDS="0 1 2 3 4"
GRAD="dnn transformer_imputed transformer_masked inr"      # noms de config
ALL_POD="lear dnn transformer_imputed transformer_masked inr"
NPAR=4
LOGDIR="logs/$(date +%m%d_%H%M)"

STAGE=${1:-all}
mkdir -p "$LOGDIR"

# ---------------------------------------------------------------------------

run_lear () {
  echo "=== LEAR (CPU) ==="
  for m in $MARKETS; do
    echo "uv run runner.py --config-name=lear dataset.name=$m"
    echo "uv run runner.py --config-name=lear dataset.name=$m model.use_lookback=false"
  done | xargs -P 5 -I{} bash -c '{}' > "$LOGDIR/lear.log" 2>&1
  echo "  -> $LOGDIR/lear.log"
}

run_grad () {
  echo "=== modeles a gradient (GPU, $NPAR en parallele) ==="
  for m in $MARKETS; do
    for cfg in $GRAD; do
      for s in $SEEDS; do
        echo "uv run runner.py --config-name=$cfg dataset.name=$m seed=$s dataset.num_workers=1"
        echo "uv run runner.py --config-name=$cfg dataset.name=$m seed=$s model.use_lookback=false dataset.num_workers=1"
      done
    done
  done | xargs -P $NPAR -I{} bash -c '{}' > "$LOGDIR/grad.log" 2>&1
  echo "  -> $LOGDIR/grad.log"
}

run_pod () {
  echo "=== balayages ==="
  for m in $MARKETS; do
    for cfg in $ALL_POD; do
      echo "  [$cfg/$m]"
      uv run partial_observability_degradation.py \
        --config-name=$cfg dataset.name=$m >> "$LOGDIR/pod.log" 2>&1 \
        || echo "    ECHEC"
    done
  done
  echo "  -> $LOGDIR/pod.log"
}

run_ablation () {
  # Ablation du conditionnement exogene, sur DE seulement.
  #   static : le code exogene est fige sur l'horizon. L'etat final du LSTM
  #            a deja parcouru les 24 pas, donc le CONTENU est identique --
  #            seule disparait la variation avec t. C'est l'ablation qui
  #            isole le mecanisme propose, et la plus defavorable a notre
  #            these.
  #   noexog : aucune covariable. Mesure leur valeur, pas celle du
  #            mecanisme ; sert de borne.
  echo "=== ablations du conditionnement (DE) ==="
  for s in $SEEDS; do
    echo "uv run runner.py --config-name=inr dataset.name=DE seed=$s model.static_exog=true exp_suffix=static dataset.num_workers=1"
    echo "uv run runner.py --config-name=inr dataset.name=DE seed=$s model.use_exog=false exp_suffix=noexog dataset.num_workers=1"
  done | xargs -P $NPAR -I{} bash -c '{}' > "$LOGDIR/ablation.log" 2>&1

  for v in "model.static_exog=true exp_suffix=static" \
           "model.use_exog=false exp_suffix=noexog"; do
    uv run partial_observability_degradation.py --config-name=inr \
      dataset.name=DE $v >> "$LOGDIR/ablation.log" 2>&1
  done
  echo "  -> $LOGDIR/ablation.log"
}

# ---------------------------------------------------------------------------

case "$STAGE" in
  train)
    run_lear & run_grad ; wait ;;
  pod)
    run_pod ;;
  ablation)
    run_ablation ;;
  all)
    # LEAR (CPU) et les reseaux (GPU) n'utilisent pas les memes ressources
    run_lear &
    LEAR_PID=$!
    run_grad
    wait $LEAR_PID
    run_pod
    run_ablation ;;
  *)
    echo "usage: $0 [all|train|pod|ablation]" ; exit 1 ;;
esac

echo
echo "=== recapitulatif ==="
echo "fichiers de resultats : $(ls results/pod/*/raw_*.csv 2>/dev/null | wc -l)"
echo "echecs                : $(grep -h -c 'Error executing job' "$LOGDIR"/*.log 2>/dev/null | paste -sd+ | bc)"

cat <<EOF

Suite :
  uv run python summarize_pod.py
  uv run python summarize_pod.py --mechanism block
  uv run python compare_ablation.py --market DE
EOF