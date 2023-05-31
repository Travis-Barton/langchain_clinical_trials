"""Test building the Zapier tool, not running it."""
import pytest

from langchain.agents.agent_toolkits.clinical_trials.base import create_clinical_trials_agent



def test_clinical_trials_agent() -> None:
    """Test that the default prompt is being inserted."""
    tool = create_clinical_trials_agent(
        llm="test",
    )

    # Test that the base prompt was successfully assigned to the default prompt
    assert tool.llm == "test"

