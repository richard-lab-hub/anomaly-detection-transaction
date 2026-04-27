"""
agent.py — AI Agent for the IEEE Fraud Detection Pipeline
---------------------------------------------------------
Uses a free local Ollama model (qwen2.5:3b by default) and connects
to mcp_server.py via SSE (HTTP) to run ML pipeline operations.

Prerequisites:
    1. Ollama running         →  ollama serve
    2. Model downloaded       →  ollama pull qwen2.5:7b
    3. MCP server running     →  python mcp_server.py   (separate terminal)
    4. Packages installed     →  pip install "mcp[cli]" ollama

Usage:
    python agent.py --query "What is the test AUC of the XGBoost model?"
    python agent.py --query "Train XGBoost in fast mode" --ollama_model  qwen2.5:7b
    python agent.py --query "..." --mcp_url http://localhost:8000/sse
"""

import asyncio
import argparse
import json
import sys

import httpx
import ollama
from mcp import ClientSession
from mcp.client.sse import sse_client

# ── Constants ─────────────────────────────────────────────────────────────────

MAXIMUM_ITERATIONS   = 15
DEFAULT_MODEL    = 'qwen2.5:7b'
DEFAULT_MCP_URL  = 'http://localhost:8000/sse'
DEFAULT_DATA_DIR = '/mnt/c/Users/richa/Desktop/AI_Credit_Project/IEEEAICreditRiskProject'

SYSTEM_PROMPT = f"""\
You are a fraud detection assistant for an IEEE credit risk project.
You control a complete ML pipeline through tools: split data, train models,
evaluate performance, score new transactions, and explain predictions.

Data directory: {DEFAULT_DATA_DIR}
Always use this exact data_dir value when calling any tool that requires it.

Rules:
- Always use tools to answer questions — never guess metric values.
- After receiving tool results, explain them clearly in plain English.
- If a required step is missing (e.g. model not trained yet), say so and offer to run it.
- Never ask the user for confirmation. Execute all required steps autonomously.
- Always include key numbers (ROC-AUC, fraud %, confusion matrix) in your final answer.
- Available models: 'xgboost' (GPU, best accuracy) or 'rf' (CPU, random forest).
- Explain that recall is a better metric than precision for fraud detection.
- For predict_transactions, if the user gives only a filename, build the full path
  by joining data_dir + '/' + filename.\
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def mcp_to_ollama_tool(mcp_tool) -> dict:
    """Convert an MCP tool definition to Ollama's OpenAI-compatible format."""
    return {
        'type': 'function',
        'function': {
            'name':        mcp_tool.name,
            'description': mcp_tool.description or '',
            'parameters':  mcp_tool.inputSchema,   # already JSON Schema
        },
    }


def tool_calls_to_dicts(tool_calls) -> list:
    """Serialise Ollama ToolCall objects to plain dicts for the message history."""
    return [
        {
            'function': {
                'name':      tc.function.name,
                'arguments': tc.function.arguments,
            }
        }
        for tc in (tool_calls or [])
    ]


def extract_result_text(mcp_result) -> str:
    """Pull the text content out of an MCP CallToolResult."""
    text = '\n'.join(c.text for c in mcp_result.content if hasattr(c, 'text'))
    return f'ERROR: {text}' if mcp_result.isError else text


# ── Agentic loop ──────────────────────────────────────────────────────────────

async def run_agent(query: str, ollama_model: str, data_dir: str, mcp_url: str) -> None:
    """
    Connect to the running MCP server, then loop:
      1. Send query + tool schemas → Ollama decides which tool to call
      2. Execute the tool via MCP server (which calls the existing ML scripts)
      3. Feed result back to Ollama
      4. Repeat until Ollama stops calling tools and gives a final answer
    """
    try:
        async with sse_client(url=mcp_url, timeout=10, sse_read_timeout=600) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                tools_response = await session.list_tools()
                ollama_tools   = [mcp_to_ollama_tool(t) for t in tools_response.tools]

                print(f'MCP server      : {mcp_url}')
                print(f'Tools available : {len(ollama_tools)}')
                print(f'Ollama model    : {ollama_model}')
                print(f'Data directory  : {data_dir}')
                print(f'\nQuery: {query}')
                print('-' * 60)

                messages = [
                    {
                        'role':    'system',
                        'content': SYSTEM_PROMPT + f'\nData directory: {data_dir}',
                    },
                    {
                        'role':    'user',
                        'content': query,
                    },
                ]

                for iteration in range(1, MAXIMUM_ITERATIONS + 1):
                    response = ollama.chat(
                        model=ollama_model,
                        messages=messages,
                        tools=ollama_tools,
                    )
                    msg = response.message

                    # No tool calls → Ollama has enough info for a final answer
                    if not msg.tool_calls:
                        print(f'\nAgent:\n{msg.content}')
                        return

                    # Append the assistant turn (with tool calls) to history
                    messages.append({
                        'role':       'assistant',
                        'content':    msg.content or '',
                        'tool_calls': tool_calls_to_dicts(msg.tool_calls),
                    })

                    # Execute each tool call through the MCP server
                    for tc in msg.tool_calls:
                        name = tc.function.name
                        args = tc.function.arguments or {}

                        print(f'\n[{iteration}] Calling → {name}')
                        print(f'     Args   : {json.dumps(args, indent=12)}')

                        mcp_result  = await session.call_tool(name, arguments=args)
                        result_text = extract_result_text(mcp_result)

                        preview = result_text[:400] + ('...' if len(result_text) > 400 else '')
                        print(f'     Result : {preview}')

                        # Feed result back — Ollama sees it on the next iteration
                        messages.append({'role': 'tool', 'content': result_text})

                print('\nAgent: Reached the maximum number of iterations.')

    except (ConnectionRefusedError, httpx.ConnectError, httpx.RemoteProtocolError) as e:
        print(f'\nError: Cannot connect to MCP server at {mcp_url}')
        print(f'Detail: {type(e).__name__}: {e}')
        print('Make sure mcp_server.py is running in another terminal:')
        print('  source myenv/bin/activate && python mcp_server.py')
        sys.exit(1)
    except Exception as e:
        print(f'\nUnexpected error: {type(e).__name__}: {e}')
        raise


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Fraud Detection AI Agent — Ollama + MCP (free, local)'
    )
    parser.add_argument(
        '--query',
        type=str,
        required=True,
        help='Natural language question. Example: "What is the test AUC?"',
    )
    parser.add_argument(
        '--ollama_model',
        type=str,
        default=DEFAULT_MODEL,
        help=f'Ollama model to use (must support tool calling). Default: {DEFAULT_MODEL}',
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default=DEFAULT_DATA_DIR,
        help='Folder containing CSV splits and model .pkl files.',
    )
    parser.add_argument(
        '--mcp_url',
        type=str,
        default=DEFAULT_MCP_URL,
        help=f'MCP server SSE endpoint. Default: {DEFAULT_MCP_URL}',
    )
    args = parser.parse_args()

    try:
        asyncio.run(run_agent(
            query=args.query,
            ollama_model=args.ollama_model,
            data_dir=args.data_dir,
            mcp_url=args.mcp_url,
        ))
    except KeyboardInterrupt:
        print('\nInterrupted.')


if __name__ == '__main__':
    main()
