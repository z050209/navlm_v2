"""Synth subpackage.

Re-exports the most-used names so callers can `from synth import ...` instead of
reaching into individual modules.
"""

from .backends import call_local_vlm, call_teacher  # noqa: F401
from .prompts import (  # noqa: F401
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    QUESTION_TEMPLATES,
    TIER_KEYWORDS,
)
from .sampling import (  # noqa: F401
    poi_tier,
    sample_destinations_weighted,
    reverse_lookup_location,
)
from .verifier import (  # noqa: F401
    parse_assistant,
    verify,
    visual_verify_poi,
)
