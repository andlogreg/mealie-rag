from unittest.mock import MagicMock

from typer.testing import CliRunner

from mealierag.cli import app, setup_logging

runner = CliRunner()


def test_setup_logging(mocker):
    """Test logging setup."""
    mock_logger = MagicMock()
    mock_get_logger = mocker.patch("logging.getLogger", return_value=mock_logger)

    mocker.patch("mealierag.cli.settings.log_level", "DEBUG")
    mocker.patch("mealierag.cli.settings.dependency_log_level", "INFO")

    setup_logging()

    assert mock_get_logger.call_count >= 2
    mock_logger.setLevel.assert_called()


def test_cli_fetch(mocker):
    """Test fetch command."""
    mock_main = mocker.patch("mealierag.cli.fetch_main")
    result = runner.invoke(app, ["fetch"])
    assert result.exit_code == 0
    mock_main.assert_called_once()


def test_cli_ingest(mocker):
    """Test ingest command."""
    mock_main = mocker.patch("mealierag.cli.ingest_main")
    result = runner.invoke(app, ["ingest"])
    assert result.exit_code == 0
    mock_main.assert_called_once()


def test_cli_qa_cli(mocker):
    """Test qa-cli command."""
    mock_main = mocker.patch("mealierag.cli.qa_cli_main")
    result = runner.invoke(app, ["qa-cli"])
    assert result.exit_code == 0
    mock_main.assert_called_once()


def test_cli_qa_ui(mocker):
    """Test qa-ui command."""
    mock_main = mocker.patch("mealierag.cli.qa_ui_main")
    result = runner.invoke(app, ["qa-ui"])
    assert result.exit_code == 0
    mock_main.assert_called_once()
