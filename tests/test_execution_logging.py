"""Tests for execution logging integration with API server."""

from __future__ import annotations

import logging
from io import StringIO
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from agentwatch.api.server import app, _execution_loggers
from agentwatch.core.schema import (
    AgentEvent,
    AgentFramework,
    AgentSession,
    EventType,
    ExecutionStatus,
    ToolCallData,
    ToolResultData,
)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_session():
    return AgentSession(
        session_id="test-session-123",
        agent_id="test-agent",
        agent_name="Test Agent",
        framework=AgentFramework.CLAUDE_CODE,
        goal="Test goal",
    )


@pytest.fixture
def cleanup_loggers():
    yield
    _execution_loggers.clear()


class TestExecutionLoggerInitialization:
    """Test that ExecutionLogger is initialized when sessions are created."""

    def test_session_creates_execution_logger(self, client, sample_session, cleanup_loggers):
        response = client.post(
            "/api/v1/sessions",
            json=sample_session.model_dump(mode="json"),
        )
        assert response.status_code == 200
        assert sample_session.session_id in _execution_loggers
        logger_inst = _execution_loggers[sample_session.session_id]
        assert logger_inst.agent_id == "test-agent"
        assert logger_inst.session_id == "test-session-123"

    def test_multiple_sessions_have_separate_loggers(
        self, client, cleanup_loggers
    ):
        session1 = AgentSession(
            session_id="session-1",
            agent_id="agent-1",
            agent_name="Agent 1",
            framework=AgentFramework.CLAUDE_CODE,
            goal="Goal 1",
        )
        session2 = AgentSession(
            session_id="session-2",
            agent_id="agent-2",
            agent_name="Agent 2",
            framework=AgentFramework.CLAUDE_CODE,
            goal="Goal 2",
        )

        client.post("/api/v1/sessions", json=session1.model_dump(mode="json"))
        client.post("/api/v1/sessions", json=session2.model_dump(mode="json"))

        assert "session-1" in _execution_loggers
        assert "session-2" in _execution_loggers
        assert _execution_loggers["session-1"].agent_id == "agent-1"
        assert _execution_loggers["session-2"].agent_id == "agent-2"


class TestExecutionLogging:
    """Test execution step logging during event ingestion."""

    def test_tool_call_logged(
        self, client, sample_session, cleanup_loggers, caplog
    ):
        client.post("/api/v1/sessions", json=sample_session.model_dump(mode="json"))

        with caplog.at_level(logging.INFO):
            event = AgentEvent(
                session_id="test-session-123",
                agent_id="test-agent",
                framework=AgentFramework.CLAUDE_CODE,
                event_type=EventType.TOOL_CALL,
                tool_call=ToolCallData(
                    tool_name="bash",
                    raw_command="ls -la",
                    arguments={"command": "ls -la"},
                ),
            )
            response = client.post(
                "/api/v1/events",
                json=event.model_dump(mode="json"),
            )
            assert response.status_code == 200

        assert "Execution step: bash" in caplog.text

    def test_tool_result_logged(
        self, client, sample_session, cleanup_loggers, caplog
    ):
        client.post("/api/v1/sessions", json=sample_session.model_dump(mode="json"))

        with caplog.at_level(logging.INFO):
            event = AgentEvent(
                session_id="test-session-123",
                agent_id="test-agent",
                framework=AgentFramework.CLAUDE_CODE,
                event_type=EventType.TOOL_RESULT,
                status=ExecutionStatus.SUCCESS,
                duration_ms=150,
                tool_result=ToolResultData(tool_name="bash", output="result output"),
            )
            response = client.post(
                "/api/v1/events",
                json=event.model_dump(mode="json"),
            )
            assert response.status_code == 200

        assert "tool_result_bash" in caplog.text
        assert "duration_ms" in caplog.text

    def test_session_end_logged(
        self, client, sample_session, cleanup_loggers, caplog
    ):
        client.post("/api/v1/sessions", json=sample_session.model_dump(mode="json"))

        with caplog.at_level(logging.INFO):
            event = AgentEvent(
                session_id="test-session-123",
                agent_id="test-agent",
                framework=AgentFramework.CLAUDE_CODE,
                event_type=EventType.SESSION_END,
                status=ExecutionStatus.SUCCESS,
                duration_ms=5000,
            )
            response = client.post(
                "/api/v1/events",
                json=event.model_dump(mode="json"),
            )
            assert response.status_code == 200

        assert "Execution complete:" in caplog.text
        assert "success" in caplog.text.lower()

    def test_non_tracked_event_type_handled_gracefully(
        self, client, sample_session, cleanup_loggers
    ):
        client.post("/api/v1/sessions", json=sample_session.model_dump(mode="json"))

        event = AgentEvent(
            session_id="test-session-123",
            agent_id="test-agent",
            framework=AgentFramework.CLAUDE_CODE,
            event_type=EventType.PLANNER_OUTPUT,
            planner_output_preview="Planning...",
        )
        response = client.post(
            "/api/v1/events",
            json=event.model_dump(mode="json"),
        )
        assert response.status_code == 200

    def test_event_without_logger_handled_gracefully(
        self, client, cleanup_loggers
    ):
        event = AgentEvent(
            session_id="unknown-session",
            agent_id="test-agent",
            framework=AgentFramework.CLAUDE_CODE,
            event_type=EventType.TOOL_CALL,
            tool_call=ToolCallData(
                tool_name="bash",
                raw_command="test",
                arguments={},
            ),
        )
        response = client.post(
            "/api/v1/events",
            json=event.model_dump(mode="json"),
        )
        assert response.status_code == 200


class TestLogFormatting:
    """Test that execution logs are properly formatted."""

    def test_execution_log_includes_context(
        self, client, sample_session, cleanup_loggers, caplog
    ):
        client.post("/api/v1/sessions", json=sample_session.model_dump(mode="json"))

        with caplog.at_level(logging.INFO):
            event = AgentEvent(
                session_id="test-session-123",
                agent_id="test-agent",
                framework=AgentFramework.CLAUDE_CODE,
                event_type=EventType.TOOL_CALL,
                tool_call=ToolCallData(
                    tool_name="bash",
                    raw_command="echo test",
                    arguments={"command": "echo test"},
                ),
            )
            client.post("/api/v1/events", json=event.model_dump(mode="json"))

        logs = [record.message for record in caplog.records]
        json_logs = [log for log in logs if log.startswith("{")]
        assert len(json_logs) > 0

        import json

        parsed = json.loads(json_logs[0])
        assert "agent_id" in parsed
        assert "session_id" in parsed
        assert "timestamp" in parsed
        assert "level" in parsed
