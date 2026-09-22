#!/usr/bin/env bash
# Campagne : modeles a gradient, balayages, ablations.
#
# LEAR n'est PAS reestime : son estimation est deterministe et ses
# checkpoints existent deja. Ses balayages restent inclus (inference seule,
# quelques minutes).
#
# Bras :
#   lear                    reference du benchmark (balayage seulement)
#   dnn                     seconde reference du benchmark
#   transformer_imputed     attention sur serie imputee
#   transformer_masked      MEME architecture, positions absentes exclues de
#                           l'attention. La paire isole le mode d'ingestion
#                           a architecture constante : memes couches, memes
#                           parametres, meme entrainement -- seule change
#                           l'ingestion.
#   inr                     le modele propose
#
# Chaque modele a gradient est aussi estime SANS lookback
# (use_lookback=false), ce qui donne la borne de rho : un modele reestime
# sans historique a realloue sa capacite sur les seules exogenes, donc il
# est correctement specifie pour cette tache. Le point a rate = 1.0, lui,
# est un modele entraine AVEC lookback puis brutalement prive de tout ; il
# opere hors de son domaine et peut diverger, ce qui en ferait une borne
# trompeuse.
#
# L'imputation est fixee a `linear`, choisie par comparaison prealable sur
# DE : en regime ponctuel elle degrade nettement moins que la saisonniere
# (LEAR +25 % contre +41 %), qui ne reprend l'avantage qu'en regime par
# blocs.
#
# Repartition GPU : si deux cartes sont visibles, la file est scindee en
# deux moities independantes, une par carte. Sinon le comportement est
# inchange.
#
# Usage :  bash run_campaign.sh              # tout
#          bash run_campaign.sh train        # estimation seule
#          bash run_campaign.sh pod          # balayages seuls
#          bash run_campaign.sh ablation     # ablations du conditionnement
#
# Variables : MARKETS, SEEDS, NPAR, GRAD, SUFFIX

set -u

MARKETS=${MARKETS:-"NP PJM BE FR DE"}
SEEDS=${SEEDS:-"0 1 2 3 4"}
GRAD=${GRAD:-"dnn transformer_imputed transformer_masked inr"}
NPAR=${NPAR:-2}
SUFFIX=${SUFFIX:-}

POD_CONFIGS="lear $GRAD"
SUF_ARG=""
[ -n "$SUFFIX" ] && SUF_ARG="exp_suffix=$SUFFIX"

LOGDIR="logs/$(date +%m%d_%H%M)"
STAGE=${1:-all}
mkdir -p "$LOGDIR"

# ---------------------------------------------------------------------------

gen_train_cmds () {
  for m in $MARKETS; do
    for cfg in $GRAD; do
      for s in $SEEDS; do
        echo "uv run runner.py --config-name=$cfg dataset.name=$m seed=$s $SUF_ARG dataset.num_workers=1"
        echo "uv run runner.py --config-name=$cfg dataset.name=$m seed=$s model.use_lookback=false $SUF_ARG dataset.num_workers=1"
      done
    done
  done
}

run_grad () {
  local ngpu
  ngpu=$(nvidia-smi -L 2>/dev/null | wc -l)
  ngpu=${ngpu:-0}

  echo "=== modeles a gradient ($ngpu GPU, $NPAR runs par carte) ==="
  echo "    configs : $GRAD"

  if [ "$ngpu" -ge 2 ]; then
    # une ligne sur deux par carte : les runs sont independants, donc
    # l'alternance suffit a equilibrer la charge
    gen_train_cmds | awk 'NR%2==1' \
      | CUDA_VISIBLE_DEVICES=0 xargs -P "$NPAR" -I{} bash -c '{}' \
      > "$LOGDIR/grad_gpu0.log" 2>&1 &
    gen_train_cmds | awk 'NR%2==0' \
      | CUDA_VISIBLE_DEVICES=1 xargs -P "$NPAR" -I{} bash -c '{}' \
      > "$LOGDIR/grad_gpu1.log" 2>&1 &
    wait
  else
    gen_train_cmds | xargs -P "$NPAR" -I{} bash -c '{}' \
      > "$LOGDIR/grad.log" 2>&1
  fi
  echo "  -> $LOGDIR/grad*.log"
}

run_pod () {
  echo "=== balayages ==="
  for m in $MARKETS; do
    for cfg in $POD_CONFIGS; do
      printf '  [%s/%s] ' "$cfg" "$m"
      if uv run partial_observability_degradation.py \
           --config-name="$cfg" dataset.name="$m" $SUF_ARG \
           >> "$LOGDIR/pod.log" 2>&1; then
        echo "ok"
      else
        echo "ECHEC"
      fi
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
  done | xargs -P "$NPAR" -I{} bash -c '{}' > "$LOGDIR/ablation.log" 2>&1

  for v in "model.static_exog=true exp_suffix=static" \
           "model.use_exog=false exp_suffix=noexog"; do
    uv run partial_observability_degradation.py --config-name=inr \
      dataset.name=DE $v >> "$LOGDIR/ablation.log" 2>&1
  done
  echo "  -> $LOGDIR/ablation.log"
}

# ---------------------------------------------------------------------------

echo "LEAR n'est pas reestime : checkpoints existants conserves."
echo

case "$STAGE" in
  train)    run_grad ;;
  pod)      run_pod ;;
  ablation) run_ablation ;;
  all)      run_grad ; run_pod ; run_ablation ;;
  *)        echo "usage: $0 [all|train|pod|ablation]" ; exit 1 ;;
esac

echo
echo "=== recapitulatif ==="
echo "fichiers de resultats : $(ls results/pod/*/raw_*.csv 2>/dev/null | wc -l)"
echo "echecs                : $(grep -h 'Error executing job' "$LOGDIR"/*.log 2>/dev/null | wc -l)"

