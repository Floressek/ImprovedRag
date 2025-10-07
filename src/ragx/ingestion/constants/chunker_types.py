from enum import Enum
class ChunkingStrategy(str, Enum):
    SEMANTIC = "semantic"
    TOKEN = "token"