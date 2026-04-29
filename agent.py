# Backward-compatible entry point — all logic lives in credit_risk.agent.agent
if __name__ == '__main__':
    from credit_risk.agent.agent import main
    main()
