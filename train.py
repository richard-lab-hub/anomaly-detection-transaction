# Backward-compatible re-export — all logic lives in credit_risk.pipeline.train
from credit_risk.pipeline.train import parse_args, train

if __name__ == '__main__':
    from credit_risk.pipeline.train import main
    main()