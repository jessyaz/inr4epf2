def optimize_and_evaluate(registry_name, n_trials=20, seeds=[0, 1, 2, 3, 4]):
    print(f"\n==================================================")
    print(f"  LANCEMENT OPTUNA : {registry_name.upper()} (~1M Params)")
    print(f"==================================================")

    config_name = None
    if os.path.exists("conf/config.yaml"):
        config_name = "config"
    elif os.path.exists("conf/main.yaml"):
        config_name = "main"

    with initialize(version_base=None, config_path="conf"):
        try:
            base_cfg = compose(config_name=config_name, overrides=[f"registry={registry_name}"])
        except Exception:
            base_cfg = compose(config_name=config_name, overrides=[f"+registry={registry_name}"])

    # Fusion propre du dictionnaire modèle dans la config globale
    if registry_name in ISO_1M_CONFIGS:
        override_cfg = OmegaConf.create(ISO_1M_CONFIGS[registry_name])
        base_cfg = OmegaConf.merge(base_cfg, override_cfg)

    with open_dict(base_cfg):
        base_cfg.registry = registry_name
        base_cfg.masking.rate = 0.0

    def objective(trial):
        cfg = base_cfg.copy()

        lr = trial.suggest_float("lr", 1e-5, 3e-4, log=True)
        wd = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
        dropout = trial.suggest_float("dropout", 0.05, 0.35, step=0.05)
        stride_train = trial.suggest_categorical("stride_train", [1, 2, 3, 6])

        with open_dict(cfg):
            if "optim" not in cfg.model:
                cfg.model.optim = {}
            cfg.model.optim.lr = lr
            cfg.model.optim.weight_decay = wd
            cfg.model.dropout = dropout
            cfg.window.stride_train = stride_train
            cfg.seed = 0

        try:
            val_mae, _ = run_single_experiment(cfg)
            return val_mae
        except Exception as e:
            print(f"[Trial Failed - {registry_name}] : {e}")
            return float("inf")

    # Recherche Optuna
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    print(f"\n[+] Meilleurs hyperparamètres pour {registry_name} à r=0.0 :")
    print(best_params)

    # Sauvegarde YAML
    best_config_dict = {
        "registry": registry_name,
        "model": {
            **ISO_1M_CONFIGS[registry_name]["model"],
            "dropout": best_params["dropout"],
            "optim": {
                "lr": best_params["lr"],
                "weight_decay": best_params["weight_decay"]
            }
        },
        "window": {
            "stride_train": best_params["stride_train"]
        }
    }

    Path("conf").mkdir(exist_ok=True)
    yaml_output_path = Path("conf") / f"best_{registry_name}.yaml"
    with open(yaml_output_path, "w") as f:
        yaml.dump(best_config_dict, f, default_flow_style=False, sort_keys=False)

    print(f"[+] Configuration optimale enregistrée dans : {yaml_output_path}")

    # Évaluation 5 seeds
    print(f"\n--- Évaluation finale sur 5 Seeds (seeds={seeds}) ---")
    val_maes, test_maes = [], []

    final_cfg = base_cfg.copy()
    with open_dict(final_cfg):
        if "optim" not in final_cfg.model:
            final_cfg.model.optim = {}
        final_cfg.model.optim.lr = best_params["lr"]
        final_cfg.model.optim.weight_decay = best_params["weight_decay"]
        final_cfg.model.dropout = best_params["dropout"]
        final_cfg.window.stride_train = best_params["stride_train"]

    for seed in seeds:
        cfg_seed = final_cfg.copy()
        with open_dict(cfg_seed):
            cfg_seed.seed = seed

        val_mae, test_mae = run_single_experiment(cfg_seed)
        val_maes.append(val_mae)
        test_maes.append(test_mae)
        print(f"  Seed {seed} | Val MAE: {val_mae:.4f} | Test MAE: {test_mae:.4f}")

    print(f"\n[RÉSULTAT FINAL {registry_name.upper()}]")
    print(f"MAE Moyenne Validation : {np.mean(val_maes):.4f} +/- {np.std(val_maes):.4f}")
    print(f"MAE Moyenne Test       : {np.mean(test_maes):.4f} +/- {np.std(test_maes):.4f}")