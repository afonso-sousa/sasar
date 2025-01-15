"""Constants used by text editing models."""

# Edit operations.
KEEP = "KEEP"
DELETE = "DELETE"
PAD_TAG = "PAD"


# Special tokens.
PAD = "[PAD]"
CLS = "[CLS]"
SEP = "[SEP]"
MASK = "[MASK]"

# Special tokens that indicate the start and end of a span of deleted tokens.
DELETE_SPAN_START = "[unused1]"
DELETE_SPAN_END = "[unused2]"

# For filtering out input tokens which are not used.
DELETED_TAGS = frozenset([DELETE, PAD_TAG, PAD])
