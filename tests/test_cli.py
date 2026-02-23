"""Tests for CLI entry point."""

from click.testing import CliRunner

from pkb import __version__
from pkb.cli import cli


class TestCLIVersion:
    def test_version_flag(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_help_flag(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "PKB" in result.output


class TestCLIVerboseFlag:
    def test_v_flag_accepted(self):
        """pkb -v --version이 정상 동작."""
        runner = CliRunner()
        result = runner.invoke(cli, ["-v", "--version"])
        assert result.exit_code == 0

    def test_vv_flag_accepted(self):
        """pkb -vv --version이 정상 동작."""
        runner = CliRunner()
        result = runner.invoke(cli, ["-vv", "--version"])
        assert result.exit_code == 0

    def test_verbose_long_flag_accepted(self):
        """pkb --verbose --version이 정상 동작."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--verbose", "--version"])
        assert result.exit_code == 0
