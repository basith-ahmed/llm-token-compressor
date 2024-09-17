import re
import logging
from typing import List, Dict, Union
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Config:
    stop_words: set = field(default_factory=lambda: {
        "the", "is", "in", "at", "of", "and", "to", "for", "on", "with", "a", "an"
    })
    synonyms: Dict[str, str] = field(default_factory=lambda: {
        "utilize": "use", "demonstrate": "show", "accomplish": "do",
        "in order to": "to", "due to the fact that": "because",
        "approximately": "approx.", "for example": "e.g.",
        "do not": "don't", "cannot": "can't", "does not": "doesn't",
        "implement": "use", "facilitate": "help", "leverage": "use",
        "optimize": "improve", "enhance": "improve", "mitigate": "reduce",
        "necessitate": "need", "utilize": "use", "commence": "start",
        "terminate": "end", "subsequent": "later", "prior to": "before",
        "in the event that": "if", "despite the fact that": "although",
        "at this point in time": "now", "in the near future": "soon"
    })
    redundant_phrases: Dict[str, str] = field(default_factory=lambda: {
        "repeat again": "repeat", "added bonus": "bonus",
        "advance planning": "planning", "basic essentials": "essentials",
        "blend together": "blend", "collaborate together": "collaborate",
        "end result": "result", "future plans": "plans",
        "past history": "history", "revert back": "revert",
        "sum total": "total", "unexpected surprise": "surprise"
    })
    number_mapping: Dict[str, str] = field(default_factory=lambda: {
        "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
        "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
        "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
        "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
        "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40",
        "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80",
        "ninety": "90", "hundred": "100", "thousand": "1000", "million": "1000000"
    })
    compression_levels: Dict[int, str] = field(default_factory=lambda: {
        1: "minimal", 2: "moderate", 3: "aggressive", 4: "maximum"
    })
    unnecessary_adjectives: set = field(default_factory=lambda: {
        "very", "extremely", "really", "just", "simply", "quite", "rather",
        "somewhat", "fairly", "pretty", "totally", "absolutely", "completely",
        "utterly", "entirely", "fully", "thoroughly", "wholly", "perfectly"
    })

class LLMTextSimplifier:
    def __init__(self, config: Config = None, level: int = 1):
        self.config = config or Config()
        self.set_level(level)

    def set_level(self, level: int) -> None:
        if level in self.config.compression_levels:
            self._level = level
        else:
            raise ValueError(f"Invalid level. Choose between {min(self.config.compression_levels.keys())} and {max(self.config.compression_levels.keys())}.")

    def simplify_sentence(self, sentence: str) -> str:
        logger.info(f"Original sentence: {sentence}")
        sentence = self._simplify(sentence)
        logger.info(f"Simplified sentence (Level {self._level}): {sentence}")
        return sentence

    def _simplify(self, sentence: str) -> str:
        tokens = self.tokenize(sentence)

        if self._level >= 1:
            tokens = self.remove_stop_words(tokens)
            tokens = self.replace_synonyms(tokens)

        if self._level >= 2:
            tokens = self.remove_unnecessary_adjectives(tokens)
            sentence = ' '.join(tokens)
            sentence = self.simplify_phrases(sentence)

        if self._level >= 3:
            sentence = self.convert_passive_to_active(sentence)
            sentence = self.remove_redundant_phrases(sentence)

        if self._level == 4:
            sentence = self.compress_max(sentence)

        sentence = self.split_long_sentences(sentence)
        sentence = self.convert_numbers(sentence)
        return sentence.strip()

    def tokenize(self, sentence: str) -> List[str]:
        return re.findall(r'\b\w+\b', sentence.lower())

    def remove_stop_words(self, tokens: List[str]) -> List[str]:
        return [word for word in tokens if word not in self.config.stop_words]

    def replace_synonyms(self, tokens: List[str]) -> List[str]:
        return [self.config.synonyms.get(word, word) for word in tokens]

    def simplify_phrases(self, sentence: str) -> str:
        for long_phrase, short_phrase in self.config.synonyms.items():
            sentence = re.sub(rf"\b{re.escape(long_phrase)}\b", short_phrase, sentence, flags=re.IGNORECASE)
        return sentence

    def remove_redundant_phrases(self, sentence: str) -> str:
        for phrase, replacement in self.config.redundant_phrases.items():
            sentence = re.sub(rf"\b{re.escape(phrase)}\b", replacement, sentence, flags=re.IGNORECASE)
        return sentence

    def convert_passive_to_active(self, sentence: str) -> str:
        passive_patterns = [
            (r"\bis\b (being )?(done|used|shown|demonstrated) by\b", "does"),
            (r"\bare\b (being )?(done|used|shown|demonstrated) by\b", "do"),
        ]
        for pattern, replacement in passive_patterns:
            sentence = re.sub(pattern, replacement, sentence, flags=re.IGNORECASE)
        return sentence

    def remove_unnecessary_adjectives(self, tokens: List[str]) -> List[str]:
        return [word for word in tokens if word not in self.config.unnecessary_adjectives]

    def convert_numbers(self, sentence: str) -> str:
        for word, num in self.config.number_mapping.items():
            sentence = re.sub(rf"\b{word}\b", num, sentence, flags=re.IGNORECASE)
        return sentence

    def compress_max(self, sentence: str) -> str:
        sentence = re.sub(r'\b(is|are|am|was|were)\b', '', sentence, flags=re.IGNORECASE)
        sentence = re.sub(r'\b(has|have|had)\b', '', sentence, flags=re.IGNORECASE)
        sentence = re.sub(r'\b(that|which|who)\b', '', sentence, flags=re.IGNORECASE)
        return sentence.strip()

    def split_long_sentences(self, sentence: str) -> str:
        if len(sentence.split()) > 20:
            parts = re.split(r'[,;]', sentence)
            if len(parts) > 1:
                sentence = '. '.join(part.strip().capitalize() for part in parts)
        return sentence

    def batch_simplify(self, sentences: List[str]) -> List[str]:
        return [self.simplify_sentence(sentence) for sentence in sentences]

def main():
    simplifier = LLMTextSimplifier()
    sentence = input()

    for level in range(1, 5):
        simplifier.set_level(level)
        simplified = simplifier.simplify_sentence(sentence)
        print(f"Level {level}: {simplified}")

if __name__ == "__main__":
    main()
