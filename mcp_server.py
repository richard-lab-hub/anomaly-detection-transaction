# Backward-compatible entry point — all logic lives in credit_risk.server.mcp_server
if __name__ == '__main__':
    from credit_risk.server.mcp_server import main
    main()
