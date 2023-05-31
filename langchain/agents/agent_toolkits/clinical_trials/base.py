"""Agent for working with calls to https://clinicaltrials.gov/api/gui."""

from typing import Any, Dict, Optional
from langchain.agents import ZeroShotAgent, Tool
from langchain.agents import load_tools
from langchain.agents.agent import AgentExecutor
from langchain.base_language import BaseLanguageModel
from langchain.chains.llm import LLMChain


def create_clinical_trials_agent(
        llm: BaseLanguageModel,
        requests_kwargs: Optional[dict] = None,
        verbose: bool = False,
        return_intermediate_steps: bool = False,
        max_iterations: Optional[int] = 15,
        max_execution_time: Optional[float] = None,
        early_stopping_method: str = "force",
        agent_executor_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
) -> AgentExecutor:
    """Create clinical trials agent by querying https://clinicaltrials.gov/api/gui."""
    try:
        import requests
    except ImportError:
        raise ValueError(
            "requests package not found, please install with `pip install requests`"
        )
    _kwargs = requests_kwargs or {}
    requests_tool = load_tools(["requests_all"])[0]
    description = """
    The ClinicalTrials.gov application programming interface (API) provides a toolbox for programmers and other technical users to use to access all posted information on ClinicalTrials.gov study records data. The API is designed for encoding simple and complex search expressions and parameters in URLs.
    There are three types of API calls:
    | Full Studies   | Retrieves all content from the first study record returned for a submitted query by default. Returns up to 100 study records per query when the minimum rank and maximum rank parameters are set in a query URL and up to 10,000 records using the Full Studies interactive demonstration. |
    | Study Fields   | Retrieves the values of one or more fields from up to all study records returned for a submitted query by default. Returns up to 1,000 study records per query when the minimum rank and maximum rank parameters are set in a query URL and up to all study records using the Study Fields interactive demonstration. |
    | Field Values   | Retrieves a unique list of values for one study field from all study records returned for a submitted query.   |

    You have access to the following fields:
    LeadSponsorName, ArmGroupInterventionName, InterventionDescription, InterventionType, InterventionName, Phase, StudyFirstSubmitDate, CompletionDate, Condition, ConditionAncestorTerm, ConditionMeshTerm, OverallStatus, OutcomeMeasureTitle, OutcomeMeasureDescription, StudyFirstSubmitQCDate, CompletionDate

    Here is an example query for https://clinicaltrials.gov/api/gui/demo/simple_study_fields:
    `https://clinicaltrials.gov/api/query/study_fields?expr=NCT01836679+OR+NCT01480011&min_rnk=1&max_rnk=1000&fmt=json&fields=nctid,ArmGroupInterventionName`
    
    Here is an example query for https://clinicaltrials.gov/api/gui/demo/simple_field_values:
    `https://clinicaltrials.gov/api/query/field_values?expr=heart+attack&field=Condition&fmt=xml`

    Here is an example query for https://clinicaltrials.gov/api/gui/demo/simple_full_study:
    `https://clinicaltrials.gov/api/query/full_studies?expr=heart+attack&min_rnk=1&max_rnk=&fmt=xml`
    """
    clinical_requests_tool = Tool(
            name='clinical_http_requests',
            func=requests_tool.run,
            description=description
        )
    llm_chain = LLMChain(
        llm=llm,
        prompt=description,
    )
    tool_names = [requests_tool.name]
    agent = ZeroShotAgent(
        llm_chain=llm_chain,
        allowed_tools=tool_names,
        **kwargs,
    )
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=[clinical_requests_tool],
        verbose=verbose,
        return_intermediate_steps=return_intermediate_steps,
        max_iterations=max_iterations,
        max_execution_time=max_execution_time,
        early_stopping_method=early_stopping_method,
        **(agent_executor_kwargs or {}),
    )
