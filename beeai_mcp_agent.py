import asyncio
import sys
import os
from typing import Any, List

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from beeai_framework.adapters.ollama import OllamaChatModel
from beeai_framework.backend import SystemMessage, UserMessage
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.mcp import MCPTool

# Load environment variables
load_dotenv()

# Create server parameters for kubernetes MCP server
server_params = StdioServerParameters(
    command="npx",
    args=["-y", "kubernetes-mcp-server@latest"],
    env={
        "PATH": os.getenv("PATH", default=""),
    },
)

async def main() -> None:
    """Main application loop"""
    namespace = input("Enter the namespace to check: ")
    if not namespace:
        namespace = "default"
        print(f"Using default namespace: {namespace}")

    async with stdio_client(server_params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()

        # Get all available tools
        all_tools = await MCPTool.from_client(session)

        # Find the tools we need
        pods_list_tool = next((t for t in all_tools if t.name == "pods_list_in_namespace"), None)
        events_tool = next((t for t in all_tools if t.name == "events_list"), None)
        pods_log_tool = next((t for t in all_tools if t.name == "pods_log"), None)

        # Step 1: Get pods in namespace
        print(f"Collecting pods in namespace '{namespace}'...")
        pods_info = "No pod information available"
        if pods_list_tool:
            result = await pods_list_tool.run({"namespace": namespace})
            pods_info = result.get_text_content()
            print(f"Retrieved pod information ({len(pods_info)} bytes)")
        else:
            print("Warning: pods_list_in_namespace tool not found")

        # Step 2: Get events
        print("Collecting events...")
        events_info = "No events information available"
        if events_tool:
            result = await events_tool.run({"namespace": namespace})
            events_info = result.get_text_content()
            print(f"Retrieved events information ({len(events_info)} bytes)")
        else:
            print("Warning: events_list tool not found")

        # Step 3: Analyze data with LLM
        print("Analyzing data with LLM...")
        llm = OllamaChatModel("granite3.1-dense:2b")
        memory = UnconstrainedMemory()

        # Add system message with instructions
        await memory.add(SystemMessage(
            "You are a Kubernetes troubleshooting expert. Analyze the pod and event information and provide a concise root cause analysis. Focus on identifying issues and suggesting fixes."
        ))

        # Add context information
        summary = f"""
Namespace: {namespace}

Pod Information:
{pods_info}

Recent Events:
{events_info}
        """

        await memory.add(UserMessage(
            f"Based on the following Kubernetes information, generate a concise root cause analysis report. Identify any issues with pods in the '{namespace}' namespace and suggest potential fixes.\n\n{summary}"
        ))

        # Generate analysis
        response = await llm.create(messages=memory.messages)
        analysis = response.get_text_content()

        # Output final report
        print("\n\n=== KUBERNETES ANALYSIS REPORT ===\n")
        print(analysis)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        print(e.explain())
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)