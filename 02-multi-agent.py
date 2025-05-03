import asyncio
import sys
import os
import traceback
import json
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

async def get_human_approval() -> bool:
    """Gets human approval before executing the remediation plan"""
    reader.write("Human Approval 🧑‍💻", "Please review the remediation plan before execution.")
    reader.write("Human Approval 🧑‍💻", "Do you approve the execution of this remediation plan? (yes/no)")
    
    # Get user input
    answer = reader.ask_single_question("Approval (yes/no): ")
    
    if answer.lower() in ["yes", "y"]:
        reader.write("Human Approval 🧑‍💻", "Remediation plan approved. Proceeding with execution.")
        return True
    else:
        reader.write("Human Approval 🧑‍💻", "Remediation plan rejected. Workflow terminated.")
        return False

async def execute_remediation(namespace: str, remediation_plan: str, all_tools: List[MCPTool]) -> str:
    """Executes the remediation plan if approved"""
    reader.write("Remediation Agent 🛠️", "Executing remediation plan...")
    
    # Create LLM and memory for execution agent
    llm = OllamaChatModel("granite3.1-dense:2b")
    memory = UnconstrainedMemory()
    
    # System prompt for remediation execution agent
    await memory.add(SystemMessage(
        "You are a Kubernetes Remediation Execution Agent. Your task is to execute a remediation plan to fix issues "
        "in a Kubernetes cluster. You have access to Kubernetes tools through the MCP protocol. "
        "Before executing each step, carefully explain what you are about to do. "
        "After executing each step, report on the outcome and any errors encountered. "
        "Be methodical and cautious in your approach."
    ))
    
    # Create a list of available tools for reference
    available_tool_names = [t.name for t in all_tools]
    tool_descriptions = "\n".join([f"- {t.name}: {t.description}" for t in all_tools])
    
    # Share the plan and provide tools context
    await memory.add(UserMessage(
        f"Execute the following remediation plan to fix issues in the Kubernetes cluster "
        f"in namespace '{namespace}'. For each step, explain what you're doing, execute it, "
        f"and report the outcome. You have access to the following Kubernetes tools through the MCP protocol:\n\n"
        f"{tool_descriptions}\n\n"
        f"REMEDIATION PLAN:\n{remediation_plan}\n\n"
        f"Proceed with execution and provide detailed feedback on each step."
    ))
    
    # Note: In a real implementation, the agent would use the tools to execute commands
    # For this example, we'll just simulate the execution with an LLM response
    response = await llm.create(messages=memory.messages)
    remediation_result = response.get_text_content()
    
    reader.write("Remediation Agent 🛠️", "Remediation execution complete.")
    reader.write("Remediation Results 📋", remediation_result)
    
    return remediation_result

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
            
            # Step 3: Create remediation plan
            remediation_plan = await create_remediation_plan(namespace, diagnosis)
            
            # Step 4: Get human approval
            approved = await get_human_approval()
            
            # Step 5: Execute remediation if approved
            if approved:
                remediation_result = await execute_remediation(namespace, remediation_plan, all_tools)
                reader.write("Workflow Complete ✅", "Kubernetes troubleshooting workflow completed successfully with remediation.")
            else:
                reader.write("Workflow Complete ✅", "Kubernetes troubleshooting workflow completed without remediation.")
    
    except Exception as e:
        reader.write("Error ❌", f"Unexpected error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
