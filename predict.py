# Backward-compatible re-export — all logic lives in credit_risk.pipeline.predict
from credit_risk.pipeline.predict import parse_args, run_shap_local, _load_threshold, predict

if __name__ == '__main__':
    from credit_risk.pipeline.predict import main
    main()
