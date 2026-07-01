.PHONY: venv

# @ execute command without printing it

venv:
	@test -d .venv || python3 -m venv .venv
	@. .venv/bin/activate && exec $$SHELL
