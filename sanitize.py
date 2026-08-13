#!/usr/bin/env python3
"""Sanitize run_oculus_workflow.py for the PUBLIC repo (08-13).

Replaces all oculus/roni/deepseek specifics with generic placeholders and
env-driven config. The private oculus copy is untouched — this runs on a
staged copy only. Order matters: longest/most-specific patterns first.
"""
import re, sys

f = 'run_oculus_workflow.py'
src = open(f, encoding='utf-8').read()
orig = src

# ── 1. full-path literals ──
src = src.replace('/home/roni/Roni_Workspace/oculus', '{REPO_DIR}')
src = src.replace('/home/roni/Roni_Workspace/audits_plans', '{WORK_DIR}')
src = src.replace('/home/roni/Roni_Workspace/tokens_keys/deepseek_api.json',
                  'LLM_API_KEY_FILE (env) or ~/.config/orchestrator/llm_api_key')
src = src.replace('/home/roni/Roni_Workspace', '~')
src = src.replace('~{REPO_DIR}', '{REPO_DIR}')  # fix accidental joining

# ── 2. filename patterns ──
src = src.replace('master_oculus_plan_', 'master_plan_')
src = src.replace('multi_agent_oculus_audit_', 'multi_agent_audit_')
src = src.replace('run_oculus_workflow.py', 'run_workflow.py')
src = src.replace('oculus_config_workflow.yaml', 'config_workflow.yaml')
src = src.replace('oculus_config.yaml', 'config.yaml')
src = src.replace('execute_master_oculus_plan.py', 'execute_master_plan.py')
src = src.replace('oculus_orchestrator.log', 'orchestrator.log')
src = src.replace('OCULUS_IMPORTANT/oculus_readme.md', 'docs/project_context.md')
src = src.replace('parallel_agent_cross_eval.py', 'parallel_agent_cross_eval.py')  # generic already
src = src.replace('.venv-orch', '.venv')

# ── 3. env var names ──
for a, b in [
    ('OCULUS_LLM_API_BASE', 'LLM_API_BASE'),
    ('OCULUS_LLM_MODEL_FLASH', 'LLM_MODEL_FAST'),
    ('OCULUS_LLM_MODEL_PRO', 'LLM_MODEL_REASONER'),
    ('OCULUS_LLM_CONTEXT_TOKENS', 'LLM_CONTEXT_TOKENS'),
    ('OCULUS_LLM_MAX_CONTEXT_TOKENS', 'LLM_MAX_CONTEXT_TOKENS'),
    ('OCULUS_LLM_CONTEXT_STEP', 'LLM_CONTEXT_STEP'),
    ('OCULUS_ON_STEP_FAILURE', 'ON_STEP_FAILURE'),
    ('OCULUS_DIR', 'REPO_DIR'),
    ('AUDITS_PLANS_DIR', 'WORK_DIR'),
    ('DEEPSEEK_API_KEY', 'LLM_API_KEY'),
    ('OCULUS_API_TOKEN', 'API_TOKEN'),
]:
    src = src.replace(a, b)

# ── 4. config key names ──
for a, b in [
    ('oculus_dir', 'repo_dir'),
    ('audits_plans_dir', 'work_dir'),
    ('deepseek_api_key', 'llm_api_key'),
    ('deepseek_api_base', 'llm_api_base'),
    ('deepseek_model', 'llm_model'),
]:
    src = src.replace(a, b)

# ── 5. class names ──
src = src.replace('OculusOrchestrator', 'MasterOrchestrator')
src = src.replace('OculusConfig', 'OrchestratorConfig')

# ── 6. model/provider strings ──
src = src.replace('"deepseek-chat"', '"your-fast-model"')
src = src.replace('"deepseek-reasoner"', '"your-reasoning-model"')
src = src.replace('https://api.deepseek.com/v1', 'https://api.your-provider.com/v1')
src = src.replace('DeepSeek V4 Pro', 'reasoning tier')
src = src.replace('DeepSeek V4 Flash', 'fast tier')
src = src.replace('DeepSeek-based', 'LLM-based')
src = src.replace('DeepSeek model', 'LLM model')
src = src.replace('DeepSeek/OmniRoute', 'your LLM provider')

# ── 7. generic oculus residue (identifiers, prompts, comments) ──
src = src.replace('oculus_X', 'project_X')            # module-style names
src = src.replace('oculus.env', 'env')                # venv exclusion
src = src.replace('oculus_env', 'env')
src = src.replace('oculus_orch_', 'orch_')            # tempfile prefix
src = re.sub(r'\boculus\.X\b', 'your_package.X', src) # shadow note
src = re.sub(r'\bOCULUS\b', 'PROJECT', src)
src = re.sub(r'\bOculus\b', 'Project', src)
src = re.sub(r'\boculus\b', 'project', src)

# ── 8. telegram chat id placeholder ──
src = src.replace('8932953349', '{TELEGRAM_CHAT_ID}')

open(f, 'w', encoding='utf-8').write(src)
print(f"changed {len(orig)} -> {len(src)} chars")
