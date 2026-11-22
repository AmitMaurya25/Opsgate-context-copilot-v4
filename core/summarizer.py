"""Lossy compression: threshold-triggered summary of the distal history by a small secondary LLM."""
from langchain_core.messages import HumanMessage, SystemMessage
from core import config
from core.llm import get_llm
from core.telemetry import count_tokens

PROMPT = """Summarize the conversation below for an agent that must continue the task. Output exactly these 5 sections:
1. Established decisions  2. Established facts  3. User preferences  4. Unsolved / pending tasks  5. Main goal and progress
Be concise. Keep facts the user already gave so they are never asked again.

<conversation>
{text}
</conversation>"""

NOTE = "[conversation so far]: "

def _text(m): return m.content if isinstance(m.content, str) else str(m.content)

def total_tokens(messages) -> int: return sum(count_tokens(_text(m)) for m in messages)

def should_summarize(messages, turn_every: int | None = None) -> bool:
    body = [m for m in messages if getattr(m, "type", "") != "system"]
    if turn_every and len([m for m in body if m.type == "human"]) % turn_every == 0 and body: return True
    return total_tokens(body) > config.SUMMARY_TRIGGER_TOKENS

def is_note(m) -> bool: return _text(m).startswith(NOTE)

def split_distal(messages, keep_last: int | None = None):
    """(system, old, recent). Old = what gets summarized; recent = last N messages kept verbatim."""
    keep_last = config.SUMMARY_KEEP_LAST_N if keep_last is None else keep_last
    system = [m for m in messages if m.type == "system"]
    body = [m for m in messages if m.type != "system"]
    if len(body) <= keep_last: return system, [], body
    return system, (body[:-keep_last] if keep_last else body), (body[-keep_last:] if keep_last else [])

def summarize(messages, keep_last: int | None = None):
    """Returns (new_messages, tokens_saved). System messages are never altered; last N are kept verbatim."""
    system, old, recent = split_distal(messages, keep_last)
    if not old: return messages, 0
    old_text = "\n".join(f"{m.type}: {_text(m)}" for m in old)
    summary = get_llm(config.SUMMARIZER_MODEL).invoke([HumanMessage(PROMPT.format(text=old_text))]).content
    note = HumanMessage(NOTE + summary)
    saved = total_tokens(old) - count_tokens(summary)
    return system + [note] + recent, saved
