from mealierag.config import SearchStrategy, Settings


def test_settings_defaults(monkeypatch):
    """
    Test default settings values when no environment variables are set.
    """
    monkeypatch.delenv("MEALIE_API_URL", raising=False)
    monkeypatch.delenv("VECTORDB_K", raising=False)
    monkeypatch.delenv("SEARCH_STRATEGY", raising=False)

    settings = Settings()
    assert settings.mealie_api_url == "http://localhost:9000/api/recipes"
    assert settings.vectordb_k == 3
    assert settings.search_strategy == SearchStrategy.SIMPLE


def test_settings_from_env(monkeypatch):
    """
    Test loading settings from environment variables.
    """
    monkeypatch.setenv("MEALIE_API_URL", "http://env-api")
    monkeypatch.setenv("VECTORDB_K", "10")
    monkeypatch.setenv("SEARCH_STRATEGY", "multiquery")

    settings = Settings()
    assert settings.mealie_api_url == "http://env-api"
    assert settings.vectordb_k == 10
    assert settings.search_strategy == SearchStrategy.MULTIQUERY
