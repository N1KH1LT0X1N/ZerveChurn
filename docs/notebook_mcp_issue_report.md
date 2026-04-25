# FastMCP Compatibility Issue Report

## Bug Description
The the-notebook-mcp package fails to start with a TypeError when using the latest version of FastMCP.

## Error Message
```
CRITICAL | the_notebook_mcp.server:server:136 (main) - Critical unexpected error in server execution: FastMCP() no longer accepts `log_level`. Pass `log_level` to `run_http_async()`, or set FASTMCP_LOG_LEVEL.

TypeError: FastMCP() no longer accepts `log_level`. Pass `log_level` to `run_http_async()`, or set FASTMCP_LOG_LEVEL.
```

## Root Cause
In `server.py` line 46 (or line 47 in v0.8.0), the code calls:
```python
mcp_server = FastMCP(..., log_level=...)
```

However, newer versions of FastMCP removed the `log_level` parameter from the `__init__` method. The parameter should now be passed to `run_http_async()` or set via the `FASTMCP_LOG_LEVEL` environment variable.

## Affected Versions
- the-notebook-mcp v0.9.0 (latest)
- the-notebook-mcp v0.8.0

## Reproduction Steps
1. Install the latest version: `uvx the-notebook-mcp start --allow-root /path/to/notebooks`
2. Server fails to start with the TypeError above

## Suggested Fix
Remove `log_level` from the FastMCP() constructor call in `server.py`. According to the FastMCP changelog, this parameter was removed and should be handled differently.

## Environment
- OS: Windows
- uvx: 0.8.3
- Python: 3.13

## How to Report
Since GitHub issues are disabled, try:
1. Contact the maintainer via email or social media
2. Create a discussion on the repository
3. Fork and submit a pull request with the fix
