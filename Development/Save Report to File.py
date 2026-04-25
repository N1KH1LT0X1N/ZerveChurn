from datetime import datetime

# Save the AI-generated report as weekly_insights_report.md
# Guard against the LLM block (`Weekly Insights Executive Briefing.text`)
# being skipped locally — `run_canvas_locally.py` does not execute Bedrock
# prompts (type=9 blocks), so `output` is undefined in a local replay. Without
# this guard the block raises NameError and breaks the run.
# See docs/repo_state_and_next_steps.md Blocker E.
if 'output' not in globals() or globals().get('output') is None:
    output = (
        "# Weekly Insights — Executive Briefing\n\n"
        "_(This report is normally produced by the Bedrock Claude-Haiku 3 LLM "
        "block `Weekly Insights Executive Briefing.text`. That block is "
        "skipped during local canvas replay because `run_canvas_locally.py` "
        "does not execute prompt-only blocks. Set up AWS Bedrock credentials "
        "and re-run this canvas inside Zerve to populate the briefing.)_\n"
    )

_report_content = output if isinstance(output, str) else str(output)

# Write to file
with open('weekly_insights_report.md', 'w', encoding='utf-8') as _f:
    _f.write(_report_content)

print(f"✅ Report saved to weekly_insights_report.md")
print(f"📄 Report length: {len(_report_content)} characters")
print(f"🕐 Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n--- REPORT PREVIEW ---")
print(_report_content[:2000])
