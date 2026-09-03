import pytest
import numpy as np

from src.state import internal_state
from src.speech.prosody import CodeFilter
from src.cognition.interaction import extract_code_blocks
from src.vision.render import FaceRenderer


def test_extract_code_blocks():
    # Test text containing code
    text = "Here is the solution:\n```python\ndef add(a, b):\n    return a + b\n```\nLet me know if you need more!"
    spoken, code, lang = extract_code_blocks(text)
    assert code == "def add(a, b):\n    return a + b"
    assert lang == "python"
    assert "```" not in spoken
    assert "def add" not in spoken
    assert "Here is the solution:" in spoken
    assert "Let me know if you need more!" in spoken

    # Test text with only code
    pure_code = "```javascript\nconsole.log('hello');\n```"
    spoken_pure, code_pure, lang_pure = extract_code_blocks(pure_code)
    assert code_pure == "console.log('hello');"
    assert lang_pure == "javascript"
    assert spoken_pure == "Here is the code on screen."

    # Test text with no code
    regular = "I love programming in Python and C++!"
    spoken_reg, code_reg, lang_reg = extract_code_blocks(regular)
    assert code_reg is None
    assert lang_reg is None
    assert spoken_reg == regular


def test_code_filter_streaming():
    internal_state.clear_active_code()
    cf = CodeFilter()

    # Stream chunks containing code block
    c1 = cf.filter_chunk("Here is how you do it: ")
    c2 = cf.filter_chunk("```python\nfor i in range(5):\n")
    c3 = cf.filter_chunk("    print(i)\n```")
    c4 = cf.filter_chunk(" That is all!")

    # Verify what speech gets enqueued for TTS
    spoken_chunks = c1 + c2 + c3 + c4
    spoken_text = " ".join(spoken_chunks)
    assert "print(i)" not in spoken_text
    assert "for i in range" not in spoken_text
    assert "Here is how you do it:" in spoken_text
    assert "That is all!" in spoken_text

    # Verify code was published to internal state for center screen display
    active = internal_state.get_active_code()
    assert active is not None
    code_body, code_lang = active
    assert "for i in range(5):" in code_body
    assert "print(i)" in code_body
    assert code_lang == "python"

    internal_state.clear_active_code()


def test_face_renderer_coding_mode():
    renderer = FaceRenderer(width=800, height=480)

    internal_state.clear_active_code()
    frame_normal = renderer.render(target_shape=(480, 800))
    assert frame_normal.shape == (480, 800, 3)

    code_snippet = "def solve():\n    return 42"
    internal_state.set_active_code(code_snippet, lang="python")

    frame_coding = renderer.render(target_shape=(480, 800))
    assert frame_coding.shape == (480, 800, 3)

    # Coding frame and normal frame must differ significantly in pixel distribution
    diff = np.abs(frame_coding.astype(int) - frame_normal.astype(int))
    assert np.mean(diff) > 1.0, "Coding frame should have distinct layout with code card"

    internal_state.clear_active_code()
    assert internal_state.get_active_code() is None
