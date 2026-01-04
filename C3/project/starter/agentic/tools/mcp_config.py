import sys
import os

SERVER_SCRIPT = os.path.join("agentic", "tools", "mcp_server.py")

# Even though the server code is identical, we can still maintain
# separate process connections to adhere to the "Multi-Server" architectural pattern.
MCP_SERVERS = {
    "subscription_service": {
        "command": sys.executable,
        "args": [SERVER_SCRIPT],
        #"transport": "stdio"
    },
}