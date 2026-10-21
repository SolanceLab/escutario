# mcp

Escutário has no standalone MCP server yet. Today it is reached through its command line (`python -m escutario.listen`, `python -m escutario.view`) and, for the House of Solance's own API, through `escutario/worker.py`. A standalone server (starts alone, talks to your own Escutário engine; address and key as settings; HTTP mode refuses to start without a token) is still to be built. Structural gap for the owner.
