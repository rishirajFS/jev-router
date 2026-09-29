import pytest

from jev_router.config import ConfigError, load_api_key, parse_env


def test_parse_env_reads_key_value_pairs():
    text = "A=1\nB=two\n"
    assert parse_env(text) == {"A": "1", "B": "two"}


def test_parse_env_ignores_comments_blank_lines_and_strips_quotes():
    text = "# comment\n\nA=\"quoted\"\nB='single'\n"
    assert parse_env(text) == {"A": "quoted", "B": "single"}


def test_load_api_key_from_file(tmp_path):
    env = tmp_path / ".env"
    env.write_text("JEV_API_KEY=abc123\n")
    assert load_api_key(env, environ={}) == "abc123"


def test_environment_variable_overrides_file(tmp_path):
    env = tmp_path / ".env"
    env.write_text("JEV_API_KEY=from_file\n")
    assert load_api_key(env, environ={"JEV_API_KEY": "from_env"}) == "from_env"


def test_load_api_key_supports_a_named_key(tmp_path):
    env = tmp_path / ".env"
    env.write_text("JEV_API_KEY=jev\nANTHROPIC_API_KEY=ant\n")
    assert load_api_key(env, environ={}, name="ANTHROPIC_API_KEY") == "ant"
    with pytest.raises(ConfigError, match="OTHER_KEY"):
        load_api_key(env, environ={}, name="OTHER_KEY")


def test_missing_key_raises_without_leaking_anything(tmp_path):
    env = tmp_path / ".env"
    env.write_text("OTHER=1\n")
    with pytest.raises(ConfigError, match="JEV_API_KEY"):
        load_api_key(env, environ={})
