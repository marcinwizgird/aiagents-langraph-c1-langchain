import asyncio
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage
from agentic.agents.agents import AgentState
import nest_asyncio

class AgentTestFramework:
    """A reusable test framework for testing agents with different tools and scenarios."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = {}

    async def test_agent(self,
                         agent_class,
                         agent_config: Dict[str, Any],
                         tools: List[Any],
                         test_scenarios: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Test an agent with given tools and scenarios.

        Args:
            agent_class: The agent class to instantiate
            agent_config: Configuration for the agent (name, system_prompt, etc.)
            tools: List of tools to provide to the agent
            test_scenarios: List of test scenarios with 'query' and 'description'

        Returns:
            Dictionary containing test results for each scenario
        """

        if self.verbose:
            print(f"\n🧪 Testing {agent_config.get('name', 'Unknown')} Agent")
            print("-" * 60)
            print(f"📋 Tools provided: {len(tools)}")
            for tool in tools:
                print(f"   - {tool.name}")
            print(f"🎯 Test scenarios: {len(test_scenarios)}")

        # Initialize agent
        try:
            agent = agent_class(tools=tools, **agent_config)
            agent_graph = agent.get_graph()
            if self.verbose:
                print("✅ Agent initialized successfully")
        except Exception as e:
            error_msg = f"❌ Failed to initialize agent: {str(e)}"
            if self.verbose:
                print(error_msg)
            return {"initialization_error": error_msg}

        # Run test scenarios
        scenario_results = {}

        for i, scenario in enumerate(test_scenarios, 1):
            scenario_name = f"scenario_{i}"
            query = scenario["query"]
            description = scenario.get("description", f"Test scenario {i}")

            if self.verbose:
                print(f"\n🔍 Scenario {i}: {description}")
                print(f"   Query: {query}")

            # Prepare test state
            test_state = {
                "messages": [HumanMessage(content=query)],
                "user_id": scenario.get("user_id"),
                "user_info": scenario.get("user_info"),
                "ticket_status": scenario.get("ticket_status"),
                "next_step": scenario.get("next_step")
            }

            try:
                # Execute agent
                result = agent_graph.invoke(test_state)

                # Extract response
                if result.get("messages"):
                    response = result["messages"][-1].content
                    scenario_results[scenario_name] = {
                        "success": True,
                        "response": response,
                        "full_result": result,
                        "description": description,
                        "query": query
                    }

                    if self.verbose:
                        print(f"   ✅ Response: {response[:150]}{'...' if len(response) > 150 else ''}")
                else:
                    scenario_results[scenario_name] = {
                        "success": False,
                        "error": "No messages in result",
                        "description": description,
                        "query": query
                    }
                    if self.verbose:
                        print("   ❌ No response messages found")

            except Exception as e:
                error_msg = str(e)
                scenario_results[scenario_name] = {
                    "success": False,
                    "error": error_msg,
                    "description": description,
                    "query": query
                }
                if self.verbose:
                    print(f"   ❌ Error: {error_msg}")

        return scenario_results

    def generate_report(self, test_name: str, results: Dict[str, Any]) -> None:
        """Generate a summary report of test results."""

        print(f"\n📊 TEST REPORT: {test_name}")
        print("=" * 60)

        if "initialization_error" in results:
            print(f"❌ Agent initialization failed: {results['initialization_error']}")
            return

        total_scenarios = len(results)
        successful_scenarios = sum(1 for r in results.values() if r.get("success", False))

        print(
            f"📈 Success Rate: {successful_scenarios}/{total_scenarios} ({successful_scenarios / total_scenarios * 100:.1f}%)")
        print("\n📝 Scenario Details:")

        for scenario_name, result in results.items():
            status = "✅" if result.get("success", False) else "❌"
            print(f"   {status} {result.get('description', scenario_name)}")
            if not result.get("success", False):
                print(f"      Error: {result.get('error', 'Unknown error')}")

        print("\n" + "=" * 60)


async def quick_agent_test(agent_class, agent_config, tools, scenarios, verbose=True):

    async def _async_test():
        framework = AgentTestFramework(verbose=verbose)
        results = await framework.test_agent(agent_class, agent_config, tools, scenarios)
        framework.generate_report(f"{agent_config.get('name', 'Agent')} Test", results)
        return results

    nest_asyncio.apply()

    res = await _async_test()

    return res
