

The Anysphere extension bundles its own Python language server, completions, and static analysis (it’s meant to replace Pylance entirely). VS Code only allows one Python language provider at a time, so Anysphere enforces uninstalling ms-python.vscode-pylance before activating.

You'll need to use this instead: https://marketplace.cursorapi.com/items/?itemName=anysphere.cursorpyright


Cursor (built on Anysphere) ships several mutually exclusive Python analysis backends:

`anysphere.python` — the new, unified Anysphere language server.

`anysphere.pyright` — an older or experimental build that wrapped Microsoft’s Pyright engine (which powers Pylance).

Uninstall this:

IntelliCode (Microsoft)
