from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Iterator, Optional

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

logger = logging.getLogger(__name__)