from mealierag.models import Recipe
from mealierag.run_fetch import main


def test_run_fetch_main(mocker, mock_settings):
    """Test the main function of run_fetch."""
    mocker.patch("mealierag.run_fetch.settings", mock_settings)

    mock_recipes = [
        Recipe(name="Recipe 1", slug="r1", description="d1"),
        Recipe(name="Recipe 2", slug="r2", description="d2"),
    ]

    mock_fetch = mocker.patch(
        "mealierag.run_fetch.fetch_full_recipes", return_value=mock_recipes
    )
    mock_logger = mocker.patch("mealierag.run_fetch.logger")

    main()

    mock_fetch.assert_called_with(
        mock_settings.mealie_api_url, mock_settings.mealie_token
    )
    mock_logger.info.assert_any_call("Successfully fetched 2 recipes.")


def test_run_fetch_main_no_recipes(mocker, mock_settings):
    """Test main function with no recipes returned."""
    mocker.patch("mealierag.run_fetch.settings", mock_settings)

    mocker.patch("mealierag.run_fetch.fetch_full_recipes", return_value=[])
    mock_logger = mocker.patch("mealierag.run_fetch.logger")

    main()

    mock_logger.info.assert_called_with("Successfully fetched 0 recipes.")
