import nltk
from nltk.tokenize import word_tokenize

# --- Ambiguous single words ---
AMBIGUOUS_WORDS = {
    # Modal verbs
    "should": "Modal verb → introduces optionality or uncertainty",
    "may": "Modal verb → introduces optionality or uncertainty",
    "could": "Modal verb → introduces optionality or uncertainty",
    "might": "Modal verb → introduces optionality or uncertainty",
    "can": "Modal verb → unclear obligation",

    # Subjective / vague quality
    "fast": "Subjective → not measurable",
    "quick": "Subjective → not measurable",
    "efficient": "Subjective → not measurable",
    "reliable": "Subjective → not measurable",
    "robust": "Subjective → not measurable",
    "scalable": "Subjective → not measurable",
    "secure": "Subjective → not measurable",
    "optimized": "Subjective → not measurable",
    "flexible": "Subjective → not measurable",
    "intuitive": "Subjective → not measurable",
    "simple": "Subjective → not measurable",
    "easy": "Subjective → not measurable",
    "smooth": "Subjective → not measurable",
    "seamless": "Subjective → not measurable",

    # Vague quantity
    "many": "Vague quantity → not defined",
    "large": "Vague quantity → not defined",
    "small": "Vague quantity → not defined",
    "several": "Vague quantity → not defined",
    "few": "Vague quantity → not defined",

    # Time vagueness
    "soon": "Vague time → not defined",
    "quickly": "Vague time → not defined",
    "immediately": "Vague time → not defined",
    "eventually": "Vague time → not defined",

    # Weak verbs
    "handle": "Weak verb → unclear action",
    "manage": "Weak verb → unclear action",
    "process": "Generic verb → lacks detail",
    "support": "Vague capability → not defined",
    "improve": "Unclear improvement → not measurable",
    "enhance": "Unclear improvement → not measurable"
}

# --- Ambiguous phrases ---
AMBIGUOUS_PHRASES = {
    "user friendly": "Subjective → not measurable",
    "user-friendly": "Subjective → not measurable",
    "easy to use": "Subjective → not measurable",
    "high performance": "Subjective → not measurable",
    "good performance": "Subjective → not measurable",
    "better performance": "Comparative → unclear baseline",
    "reasonable time": "Vague → not defined",
    "acceptable time": "Vague → not defined",
    "as soon as possible": "Vague deadline",
    "when required": "Vague condition",
    "if necessary": "Vague condition",
    "as needed": "Vague condition",
    "at all times": "Unrealistic / vague",
    "most of the time": "Vague availability",
    "under normal conditions": "Undefined condition",
    "large number of users": "Vague quantity",
    "appropriate response": "Subjective",
    "properly": "Subjective → unclear",
    "correctly": "Subjective → unclear"
}

# --- Suggestions ---
SUGGESTIONS = {
    # Modal
    "should": "Replace with 'must' for mandatory requirement",
    "may": "Define condition or use 'must'",
    "could": "Clarify requirement or remove uncertainty",
    "might": "Avoid uncertainty; define exact behavior",
    "can": "Clarify whether it is mandatory or optional",

    # Performance
    "fast": "Specify time (e.g., within 2 seconds)",
    "quick": "Define exact response time",
    "quickly": "Specify measurable duration",
    "efficient": "Define performance metrics",

    # Quality
    "reliable": "Define uptime (e.g., 99.9%)",
    "robust": "Specify error handling conditions",
    "secure": "Specify security standards (e.g., AES, HTTPS)",
    "scalable": "Define max users/load",

    # Usability
    "user friendly": "Define usability metrics",
    "user-friendly": "Define usability metrics",
    "easy to use": "Specify usability criteria",

    # Quantity
    "many": "Specify exact number",
    "large": "Define numeric threshold",
    "several": "Specify exact count",

    # Time
    "soon": "Specify exact time",
    "immediately": "Define exact response time",
    "eventually": "Replace with deadline",

    # Conditions
    "when required": "Define exact trigger condition",
    "if necessary": "Specify when it is necessary",
    "as needed": "Define condition explicitly",

    # Weak verbs
    "handle": "Specify exact behavior",
    "manage": "Define specific operations",
    "support": "Define supported features clearly",
    "improve": "Define measurable improvement",
    "enhance": "Specify what is improved",

    # Generic
    "properly": "Define expected behavior",
    "correctly": "Specify validation rules"
}

# Exclude false positives for "ly"
EXCLUDED_LY_WORDS = ["friendly", "family", "only", "likely"]

# ---------------- MAIN FUNCTION ----------------
def detect_ambiguity(sentence):
    sentence_lower = sentence.lower()
    ambiguous = []
    seen = set()

    # Phrase detection
    for phrase, reason in AMBIGUOUS_PHRASES.items():
        if phrase in sentence_lower and phrase not in seen:
            suggestion = SUGGESTIONS.get(phrase, "Provide more specific wording")
            ambiguous.append((phrase, reason, suggestion))
            seen.add(phrase)

    # Word detection
    tokens = word_tokenize(sentence_lower)

    for word in tokens:

        if word in AMBIGUOUS_WORDS and word not in seen:
            suggestion = SUGGESTIONS.get(word, "Provide more specific wording")
            ambiguous.append((word, AMBIGUOUS_WORDS[word], suggestion))
            seen.add(word)

        elif (
            word.endswith("ly")
            and len(word) > 4
            and word not in seen
            and word not in EXCLUDED_LY_WORDS
            and "-" not in word #“We refined our linguistic rules to correctly handle
                                # compound adjectives like ‘user-friendly’ and avoid incorrect pattern-based detection.”
        ):
            reason = "Vague adverb → unclear manner"
            suggestion = "Replace with measurable or specific description"
            ambiguous.append((word, reason, suggestion))
            seen.add(word)

    return ambiguous


# ---------------- HIGHLIGHT FUNCTION ----------------
def highlight_sentence(sentence, ambiguous_items):
    highlighted = sentence

    for term, _, _ in ambiguous_items:
        highlighted = highlighted.replace(
            term,
            f"<span style='color:red; font-weight:bold'>{term}</span>"
        )

    return highlighted