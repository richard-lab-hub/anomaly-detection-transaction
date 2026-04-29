# Backward-compatible re-export — all logic lives in credit_risk.pipeline.evaluate
from credit_risk.pipeline.evaluate import parse_args, run_shap, test

if __name__ == '__main__':
    from credit_risk.pipeline.evaluate import main
    main()
