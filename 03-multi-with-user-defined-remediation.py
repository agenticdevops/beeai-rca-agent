import asyncio
import sys
import os
import traceback
import json
import re
from typing import Dict, List, Optional

# Custom console output helper
class ConsoleReader:
    def write(self, role: str, data: str) -> None:
        """Print formatted output with colored role prefix"""
        print(f"\033[1;36m{role}:\033[0m {data}")
        
    def ask_single_question(self, query_message: str) -> str:
        """Ask the user a single question and return their response"""
        try:
            answer = input(f"\033[1;33m{query_message}\033[0m ")
            return answer.strip()
        except (EOFError, KeyboardInterrupt):
            print()
            exit()
    
    def multi_line_input(self, prompt: str) -> str:
        """Get multi-line input from the user, ending with a line containing only 'END'"""
        print(f"\033[1;33m{prompt}\033[0m")
        print("Enter your instructions (type 'END' on a new line when finished):")
        lines = []
        while True:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)
        return "\n".join(lines)

# Try to import required libraries
try:
    from dotenv import load_dotenv
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from beeai_framework.adapters.ollama import OllamaChatModel
    from beeai_framework.backend import SystemMessage, UserMessage
    from beeai_framework.memory import UnconstrainedMemory
    from beeai_framework.tools.mcp import MCPTool
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please install the required packages with:")
    print("pip install beeai-framework python-dotenv mcp")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Create console reader for user interaction
reader = ConsoleReader()

# Create server parameters for kubernetes MCP server
server_params = StdioServerParameters(
    command="npx",
    args=["-y", "kubernetes-mcp-server@latest"],
    env={
        "PATH": os.getenv("PATH", default=""),
    },
)

async def collect_cluster_data(namespace: str, all_tools: List[MCPTool]) -> Dict:
    """Collects relevant data from the Kubernetes cluster"""
    reader.write("Data Collection 📊", f"Gathering information from namespace '{namespace}'...")
    
    data = {
        "namespace": namespace,
        "pods_info": "",
        "events_info": "",
        "pod_logs": {}
    }
    
    # Find the tools we need
    pods_list_tool = next((t for t in all_tools if t.name == "pods_list_in_namespace"), None)
    events_tool = next((t for t in all_tools if t.name == "events_list"), None)
    pods_log_tool = next((t for t in all_tools if t.name == "pods_log"), None)
    
    # Get pods in namespace
    if pods_list_tool:
        result = await pods_list_tool.run({"namespace": namespace})
        data["pods_info"] = result.get_text_content()
        reader.write("Data Collection 📊", f"Retrieved pod information ({len(data['pods_info'])} bytes)")
        
        # Parse pod names to get logs
        try:
            pods_data = json.loads(data["pods_info"])
            if pods_log_tool:
                for item in pods_data.get("items", []):
                    pod_name = item.get("metadata", {}).get("name", "")
                    if pod_name:
                        try:
                            log_result = await pods_log_tool.run({
                                "name": pod_name,
                                "namespace": namespace
                            })
                            data["pod_logs"][pod_name] = log_result.get_text_content()
                            reader.write("Data Collection 📊", f"Retrieved logs for pod '{pod_name}'")
                        except Exception as e:
                            reader.write("Data Collection 📊", f"Could not retrieve logs for pod '{pod_name}': {str(e)}")
        except Exception as e:
            reader.write("Data Collection 📊", f"Error parsing pod information: {str(e)}")
    else:
        reader.write("Data Collection 📊", "Warning: pods_list_in_namespace tool not found")
    
    # Get events
    if events_tool:
        result = await events_tool.run({"namespace": namespace})
        data["events_info"] = result.get_text_content()
        reader.write("Data Collection 📊", f"Retrieved events information ({len(data['events_info'])} bytes)")
    else:
        reader.write("Data Collection 📊", "Warning: events_list tool not found")
    
    return data

async def diagnose_issues(data: Dict) -> str:
    """Uses an LLM to diagnose issues in the Kubernetes cluster"""
    reader.write("Diagnosis Agent 🔍", "Analyzing cluster data to identify issues...")
    
    # Create LLM and memory
    llm = OllamaChatModel("granite3.1-dense:2b")
    memory = UnconstrainedMemory()
    
    # Prepare combined log information (truncate if too large)
    combined_logs = ""
    for pod_name, log_content in data["pod_logs"].items():
        # Take only first 1000 chars to avoid context overflow
        log_excerpt = log_content[:1000] + "..." if len(log_content) > 1000 else log_content
        combined_logs += f"\n\n--- Logs for pod {pod_name} ---\n{log_excerpt}"
    
    # System prompt for diagnosis agent
    await memory.add(SystemMessage(
        "You are a Kubernetes Diagnosis Expert. Your task is to carefully analyze pod information, events, and logs from a Kubernetes cluster. "
        "Identify any issues, errors, or warnings. Look for failed pods, error messages, resource constraints, configuration issues, "
        "or any other problems that could impact application functionality. "
        "Focus on providing a clear, detailed diagnosis of what's wrong, but do NOT suggest solutions yet. "
        "Your diagnosis will be used by another agent to create a remediation plan."
    ))
    
    # Prepare context information
    context = f"""
# Kubernetes Cluster Information
Namespace: {data["namespace"]}

# Pod Information
{data["pods_info"]}

# Recent Events
{data["events_info"]}

# Pod Logs
{combined_logs}
    """
    
    await memory.add(UserMessage(
        "Analyze the following Kubernetes information and provide a comprehensive diagnosis of any issues detected. "
        "Focus on identifying problems, error patterns, and the affected components. "
        "DO NOT provide remediation suggestions yet - just focus on the diagnosis.\n\n" + context
    ))
    
    # Generate diagnosis
    response = await llm.create(messages=memory.messages)
    diagnosis = response.get_text_content()
    
    reader.write("Diagnosis Agent 🔍", "Diagnosis complete.")
    reader.write("Diagnosis Report 📋", diagnosis)
    
    return diagnosis

async def create_remediation_plan(namespace: str, diagnosis: str) -> str:
    """Creates a remediation plan based on the diagnosis"""
    reader.write("Remediation Planner 📝", "Developing remediation plan based on diagnosis...")
    
    # Create LLM and memory
    llm = OllamaChatModel("granite3.1-dense:2b")
    memory = UnconstrainedMemory()
    
    # System prompt for remediation planner
    await memory.add(SystemMessage(
        "You are a Kubernetes Remediation Expert. Your task is to develop a detailed, step-by-step remediation plan "
        "based on a diagnosis of Kubernetes cluster issues. For each issue identified in the diagnosis, "
        "provide specific commands or actions to resolve the problem. "
        "Your plan should be specific, executable, and include Kubernetes commands where appropriate. "
        "Format your plan as a numbered list of steps. Include required kubectl commands, YAML modifications, "
        "or other specific actions. Consider potential risks and mitigation strategies."
    ))
    
    # Share diagnosis and context
    await memory.add(UserMessage(
        f"Based on the following diagnosis of our Kubernetes cluster in namespace '{namespace}', "
        f"create a detailed remediation plan that addresses all issues identified. "
        f"Include specific kubectl commands, YAML changes, or other actions needed to resolve each problem.\n\n"
        f"DIAGNOSIS:\n{diagnosis}"
    ))
    
    # Generate remediation plan
    response = await llm.create(messages=memory.messages)
    remediation_plan = response.get_text_content()
    
    reader.write("Remediation Planner 📝", "Remediation plan complete.")
    reader.write("Remediation Plan 📋", remediation_plan)
    
    return remediation_plan

async def user_guided_remediation(namespace: str, diagnosis: str, remediation_plan: str, all_tools: List[MCPTool]) -> str:
    """Gets remediation instructions from the user and executes them with LLM assistance"""
    reader.write("User-Guided Remediation 🛠️", "The diagnosis has been completed and a remediation plan suggested.")
    reader.write("User-Guided Remediation 🛠️", "You can now provide your own remediation instructions.")
    reader.write("User-Guided Remediation 🛠️", "The LLM will interpret your instructions and use Kubernetes tools to execute them.")
    
    # Display suggested remediation plan for reference
    reader.write("Suggested Remediation Plan 📋", remediation_plan)
    
    # Get user instructions
    user_instructions = reader.multi_line_input("Please provide your remediation instructions:")
    
    if not user_instructions.strip():
        reader.write("User-Guided Remediation 🛠️", "No instructions provided. Aborting remediation.")
        return "Remediation aborted due to empty instructions."
    
    # Create LLM and memory for execution agent
    llm = OllamaChatModel("granite3.1-dense:2b")
    memory = UnconstrainedMemory()
    
    # Create dictionaries of tools by name for easy access
    tool_dict = {tool.name: tool for tool in all_tools}
    
    # System prompt for execution planning agent
    await memory.add(SystemMessage(
        "You are a Kubernetes Remediation Execution Agent. Your task is to translate the user's remediation instructions "
        "into specific Kubernetes MCP tool calls. You have access to Kubernetes tools through the MCP protocol. "
        "For each instruction, provide:\n"
        "1. A brief explanation of what you're about to do\n"
        "2. The specific MCP tool to use (choose from the available tools list)\n"
        "3. The exact parameters for the tool call in JSON format\n\n"
        "Format each tool action as: [[TOOL_NAME:PARAMETERS_AS_JSON]]\n"
        "Example: [[pods_delete:{\"name\":\"broken-pod\",\"namespace\":\"default\"}]]\n\n"
        "Only include tools that are in the available tools list. Be precise with parameter names and values."
    ))
    
    # Create a list of available tools for reference
    available_tool_names = [t.name for t in all_tools]
    tool_descriptions = "\n".join([f"- {t.name}: {t.description}" for t in all_tools])
    
    # Share the user instructions, diagnosis, and tools context
    await memory.add(UserMessage(
        f"Translate these user-provided remediation instructions into specific Kubernetes MCP tool calls. "
        f"The instructions are for fixing issues in namespace '{namespace}'.\n\n"
        f"AVAILABLE TOOLS:\n{tool_descriptions}\n\n"
        f"DIAGNOSIS:\n{diagnosis}\n\n"
        f"USER INSTRUCTIONS:\n{user_instructions}\n\n"
        f"For each instruction, provide your planned action and the tool call in the format: "
        f"[[TOOL_NAME:PARAMETERS_AS_JSON]]"
    ))
    
    # Generate execution plan
    response = await llm.create(messages=memory.messages)
    execution_plan = response.get_text_content()
    
    reader.write("Execution Plan 📋", execution_plan)
    
    # Parse execution plan and execute commands
    results = []
    
    # Use regex to find all tool calls
    tool_calls = re.findall(r'\[\[(.*?):(.*?)\]\]', execution_plan)
    
    if not tool_calls:
        reader.write("Execution 🛠️", "No valid tool calls found in execution plan. Please check your instructions.")
        return "No actions executed."
    
    # Execute each tool call
    for tool_name, params_str in tool_calls:
        tool_name = tool_name.strip()
        if tool_name not in tool_dict:
            results.append(f"Error: Tool '{tool_name}' not found in available tools.")
            reader.write("Execution Error ❌", f"Tool '{tool_name}' not found in available tools.")
            continue
        
        try:
            # Parse parameters as JSON
            params = json.loads(params_str)
            reader.write("Executing 🛠️", f"Tool: {tool_name} with parameters: {params}")
            
            # Execute the tool
            result = await tool_dict[tool_name].run(params)
            result_text = result.get_text_content()
            
            # Truncate result if too long
            if len(result_text) > 500:
                result_text = result_text[:500] + "... [truncated]"
                
            results.append(f"Successfully executed {tool_name}: {result_text}")
            reader.write("Execution Success ✅", f"Completed {tool_name}")
            reader.write("Result 📊", result_text[:500] + ("..." if len(result_text) > 500 else ""))
            
        except json.JSONDecodeError:
            results.append(f"Error: Invalid JSON parameters for tool '{tool_name}': {params_str}")
            reader.write("Execution Error ❌", f"Invalid JSON parameters for tool '{tool_name}': {params_str}")
        except Exception as e:
            results.append(f"Error executing {tool_name}: {str(e)}")
            reader.write("Execution Error ❌", f"Error executing {tool_name}: {str(e)}")
    
    # Create final summary
    execution_summary = "\n\n".join(results)
    reader.write("Remediation Execution Complete 🏁", f"Executed {len(tool_calls)} tool calls")
    
    return execution_summary

async def main() -> None:
    """Main application flow"""
    try:
        reader.write("Kubernetes Multi-Agent Troubleshooter 🤖", "Starting Kubernetes troubleshooting workflow...")
        
        # Get namespace from user
        namespace = reader.ask_single_question("Enter the namespace to check [default]: ")
        if not namespace:
            namespace = "default"
            reader.write("System", f"Using default namespace: {namespace}")
        
        async with stdio_client(server_params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            
            # Get all available K8s tools
            all_tools = await MCPTool.from_client(session)
            reader.write("System", f"Connected to Kubernetes MCP server with {len(all_tools)} available tools")
            
            # Step 1: Collect data
            data = await collect_cluster_data(namespace, all_tools)
            
            # Step 2: Diagnose issues
            diagnosis = await diagnose_issues(data)
            
            # Step 3: Create remediation plan (for reference only)
            remediation_plan = await create_remediation_plan(namespace, diagnosis)
            
            # Step 4: Ask if user wants to proceed with remediation
            proceed = reader.ask_single_question("Do you want to proceed with remediation? (yes/no): ")
            
            # Step 5: Execute user-guided remediation if approved
            if proceed.lower() in ["yes", "y"]:
                remediation_result = await user_guided_remediation(namespace, diagnosis, remediation_plan, all_tools)
                reader.write("Workflow Complete ✅", "Kubernetes troubleshooting workflow completed with user-guided remediation.")
            else:
                reader.write("Workflow Complete ✅", "Kubernetes troubleshooting workflow completed without remediation.")
    
    except Exception as e:
        reader.write("Error ❌", f"Unexpected error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
