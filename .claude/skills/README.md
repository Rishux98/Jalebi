# Claude Code skills for ODYSSEY

Claude Code loads these automatically from `.claude/skills/` at session start. Restart the session after adding or editing one.

| Skill | Source | Triggers when |
|---|---|---|
| `odyssey-phase` | Written for Odyssey | You run a ROADMAP phase ("Phase P3", or paste a phase prompt). It covers plan, implementation, tests, Verify and the summary with a commit message. |
| `numpy-grid-algorithms` | Written for Odyssey | You write grid code in `terrain/`, `costmap/`, `planner/`, `belief/` or `world/`. It covers array conventions, the A* design, edge cases and the 2 s budget. |
| `explainable-decisions` | Written for Odyssey | You write autonomy, the decision log, target reasons or the dashboard's Decision Log tab. It covers the LogEntry evidence contract. |
| `webapp-testing` | [anthropics/skills](https://github.com/anthropics/skills/tree/main/skills/webapp-testing) | You test the Streamlit dashboard in a browser (P5/P6 verify). It uses Playwright. |
| `frontend-design` | [anthropics/skills](https://github.com/anthropics/skills/tree/main/skills/frontend-design) | You design or restyle the dashboard UI (P5). |
| `claude-md-improver` | [anthropics/claude-plugins-official](https://github.com/anthropics/claude-plugins-official/tree/main/plugins/claude-md-management/skills/claude-md-improver) | You ask to audit or update `CLAUDE.md`. |

## Requirements
- `webapp-testing` needs Playwright, installed once before P5:
  `pip install playwright && playwright install chromium`.
  Its only script, `scripts/with_server.py`, uses the standard library to start the app and poll a localhost port.
  Example: `python .claude/skills/webapp-testing/scripts/with_server.py --server "streamlit run app.py --server.headless true" --port 8501 -- python tests/ui_clickthrough.py`
- The other skills are instructions only. They have no scripts, network access or API keys.

Official skills keep their original `LICENSE.txt` files.
