import logging

from browser_use import Agent, BrowserSession, Tools

from optexity.inference.infra.browser import Browser
from optexity.inference.models import normalize_model
from optexity.inference.models.chat_litellm import build_agent_llm
from optexity.schema.actions.interaction_action import (
    AgenticTask,
    CloseOverlayPopupAction,
)
from optexity.schema.memory import Memory
from optexity.schema.task import Task

logger = logging.getLogger(__name__)


async def handle_agentic_task(
    agentic_task_action: AgenticTask | CloseOverlayPopupAction,
    task: Task,
    memory: Memory,
    browser: Browser,
    *,
    recorder=None,
    prompt=None,
    max_steps=None,
):
    if (
        recorder is None
        and agentic_task_action.learning
        and agentic_task_action.learning.mode != "off"
    ):
        from optexity.inference.core.interaction.action_memory import run_learned_task

        return await run_learned_task(
            agentic_task_action, task, memory, browser, handle_agentic_task
        )

    if agentic_task_action.backend == "browser_use":

        options = {}
        if recorder is not None:
            options.update(
                action_execution_callback=recorder,
                sensitive_data=recorder.secrets(),
                directly_open_url=False,
                extend_system_message=(
                    "Preserve every <secret> parameter reference verbatim in tool arguments. "
                    "Do not replace references with guessed or copied literal values."
                ),
            )

        if isinstance(agentic_task_action, CloseOverlayPopupAction):
            tools = Tools(
                exclude_actions=[
                    "search",
                    "navigate",
                    "go_back",
                    "upload_file",
                    "scroll",
                    "find_text",
                    "send_keys",
                    "evaluate",
                    "switch",
                    "close",
                    "extract",
                    "dropdown_options",
                    "select_dropdown",
                    "write_file",
                    "read_file",
                    "replace_file",
                ]
            )
        else:
            tools = Tools()
        llm = build_agent_llm(normalize_model(task.llm_provider, task.llm_model_name))
        browser_session = BrowserSession(
            cdp_url=browser.cdp_url, keep_alive=agentic_task_action.keep_alive
        )

        step_directory = (
            task.logs_directory / f"step_{str(memory.automation_state.step_index)}"
        )
        step_directory.mkdir(parents=True, exist_ok=True)

        agent = Agent(
            task=prompt if prompt is not None else agentic_task_action.task,
            llm=llm,
            browser_session=browser_session,
            use_vision=agentic_task_action.use_vision,
            tools=tools,
            calculate_cost=True,
            # Learned runs write sanitized action evidence, never raw prompts
            # containing resolved parameters or credentials.
            save_conversation_path=None if recorder is not None else step_directory,
            **options,
        )
        logger.debug(f"Starting browser session for agentic task {browser.cdp_url} ")
        try:
            await agent.browser_session.start()
            logger.debug(f"Finally running agentic task on browser_use {browser.cdp_url} ")
            history = await agent.run(
                max_steps=max_steps if max_steps is not None else agentic_task_action.max_steps
            )
            logger.debug(f"Agentic task completed on browser_use {browser.cdp_url} ")
        finally:
            if recorder is not None:
                recorder.failed = recorder.failed or getattr(
                    agent, "action_recording_failed", False
                )
                recorder.usage = getattr(agent.history, "usage", None)
                recorder.close()
            agent.stop()
            if agent.browser_session:
                await agent.browser_session.stop()
                await agent.browser_session.reset()

        return history

    elif agentic_task_action.backend == "browserbase":
        raise NotImplementedError("Browserbase is not supported yet")

    return None
