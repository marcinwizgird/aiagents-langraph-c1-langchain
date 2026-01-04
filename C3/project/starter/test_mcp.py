import asyncio
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools

from agentic.tools.mcp_config import MCP_SERVERS
from agentic.tools.mcp_server import mcp


# UPDATE: Import from the adapter package
server_params = StdioServerParameters(
    command=sys.executable,
    args=["mcp_server.py"],
)

# Ensure we can import from local modules
#sys.path.append(os.getcwd())

async def test_mcp_connection():
    print("🚀 Starting MCP Client Test (via langchain-mcp-adapters)...")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get tools
            tools = await load_mcp_tools(session)

            try:

                print("\n🔌 Connecting to server...")
                print("\n--- Testing Subscription Service ---")
                for t in tools:
                    print(f"   - {t.name}: {t.description[:50]}...")

                assert any(t.name == "lookup_customer" for t in tools), "❌ Missing 'lookup_customer'!"

                print("\n--- Testing Reservation Service ---")
                assert any(t.name == "get_available_experiences" for t in tools), "❌ Missing 'get_available_experiences'!"

                print("\n--- Testing Knowledge Service ---")
                assert any(t.name == "search_knowledge_base" for t in tools), "❌ Missing 'search_knowledge_base'!"

            except Exception as e:
                print(f"\n❌ Test Failed: {e}")
                import traceback
                traceback.print_exc()
            finally:
                print("\n🔌 Closing connections...")
                #await client.__aexit__(None, None, None)
                print("✅ Done.")


if __name__ == "__main__":
    #if sys.platform == 'win32':
    #    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(test_mcp_connection())

