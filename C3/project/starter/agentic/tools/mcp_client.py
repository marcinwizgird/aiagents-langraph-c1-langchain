from contextlib import AsyncExitStack
from typing import Dict, Any, List
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_core.tools import StructuredTool


class MultiServerMCPClient:
    def __init__(self, connections: Dict[str, Dict[str, Any]]):
        self.connections = connections
        self.exit_stack = AsyncExitStack()
        self.sessions: Dict[str, ClientSession] = {}

    async def __aenter__(self):
        for name, config in self.connections.items():
            server_params = StdioServerParameters(
                command=config["command"],
                args=config["args"],
                env=None
            )
            read, write = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.sessions[name] = session
            print(f"🔌 Connected to MCP Server: {name}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.exit_stack.aclose()

    async def get_tools(self, server_name: str) -> List[StructuredTool]:
        """Get tools specifically from one connected server."""
        if server_name not in self.sessions:
            return []

        session = self.sessions[server_name]
        result = await session.list_tools()
        tools = []
        for tool_def in result.tools:
            tools.append(self._create_langchain_tool(tool_def, session))
        return tools

    def _create_langchain_tool(self, mcp_tool, session):
        async def _wrapper(**kwargs):
            result = await session.call_tool(mcp_tool.name, arguments=kwargs)
            return "\n".join([c.text for c in result.content if c.type == "text"])

        return StructuredTool.from_function(
            func=None,
            coroutine=_wrapper,
            name=mcp_tool.name,
            description=mcp_tool.description
        )