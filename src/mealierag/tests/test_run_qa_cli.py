from unittest.mock import MagicMock

import pytest
from qdrant_client.http.models import ScoredPoint

from mealierag.models import QueryExtraction
from mealierag.run_qa_cli import main


def test_run_qa_cli_health_check_failure(mocker):
    """Test exit on health check failure."""
    mock_service = MagicMock()
    mock_service.check_health.return_value = False
    mocker.patch("mealierag.run_qa_cli.service", mock_service)

    with pytest.raises(SystemExit):
        main()


def test_run_qa_cli_loop(mocker):
    """Test CLI loop execution."""
    mock_service = MagicMock()
    mock_service.check_health.return_value = True
    mock_service.generate_queries.return_value = QueryExtraction(
        expanded_queries=["query"]
    )
    mock_service.retrieve_recipes.return_value = [
        ScoredPoint(
            id=1,
            version=1,
            score=1.0,
            payload={
                "name": "Recipe",
                "rating": 5,
                "tags": [],
                "category": "Test",
                "recipe_id": "123",
            },
        )
    ]
    mock_service.populate_messages.return_value = []

    mock_service.chat.return_value = iter(["Response"])

    mocker.patch("mealierag.run_qa_cli.service", mock_service)

    mocker.patch("builtins.input", side_effect=["Who am I?", "exit"])

    main()

    mock_service.generate_queries.assert_called_once()
    mock_service.retrieve_recipes.assert_called_once()
    mock_service.chat.assert_called_once()
