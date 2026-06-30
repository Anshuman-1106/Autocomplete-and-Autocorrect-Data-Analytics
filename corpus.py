"""
Sample text corpus for the autocomplete / autocorrect project.

This module supplies a diverse, original (non-copyrighted) collection of
sentences across several domains (technology, weather, food, sports,
travel, business, science, daily life). It exists so the project runs
out-of-the-box without external downloads. In a production project you
would swap this out for a large real-world dataset (e.g. chat logs,
emails, books, or a corpus you collect yourself) -- see
DatasetCollector.load_from_file() in preprocessing.py for that path.
"""

import random

_TOPIC_SENTENCES = {
    "technology": [
        "The new software update improves battery life significantly.",
        "Artificial intelligence is transforming how we write and read text.",
        "Developers are testing the application before the next release.",
        "The server crashed because of a memory leak in the application.",
        "Cloud computing allows companies to scale their infrastructure quickly.",
        "She wrote a python script to automate the daily report generation.",
        "The keyboard shortcut saves a lot of time during development.",
        "Machine learning models require large amounts of training data.",
        "The mobile application received great reviews from early users.",
        "Engineers are debugging the network connection issue this morning.",
        "The database query returned an unexpected result yesterday.",
        "Open source projects depend heavily on community contributions.",
    ],
    "weather": [
        "The weather forecast predicts heavy rain for the weekend.",
        "It was a sunny morning with a light breeze from the coast.",
        "The temperature dropped quickly after the sun went down.",
        "A storm warning was issued for the coastal region tonight.",
        "Clear skies are expected throughout the entire afternoon.",
        "The humidity made the evening feel warmer than usual.",
        "Snow is forecast for the mountains later this week.",
        "The wind picked up suddenly before the thunderstorm arrived.",
    ],
    "food": [
        "She added fresh basil to the tomato sauce before serving.",
        "The recipe calls for two cups of flour and a pinch of salt.",
        "He ordered a coffee and a croissant at the local cafe.",
        "The restaurant is known for its spicy noodle soup.",
        "Fresh vegetables make the salad taste much better.",
        "The bakery sells warm bread every morning before sunrise.",
        "They grilled chicken and vegetables for the family dinner.",
        "A good cup of tea can make the morning feel calmer.",
    ],
    "sports": [
        "The team practiced hard before the championship game.",
        "She trained every morning to prepare for the marathon.",
        "The coach explained the new strategy during halftime.",
        "Fans cheered loudly when the final goal was scored.",
        "He improved his running speed after months of training.",
        "The players celebrated their victory on the field.",
        "The tournament schedule was announced earlier this week.",
    ],
    "travel": [
        "They booked a flight to explore the old city center.",
        "The train was delayed because of the heavy traffic.",
        "She packed her bags the night before the long trip.",
        "The hotel offered a beautiful view of the mountains.",
        "We visited the museum and the market on the same day.",
        "The airport was crowded during the holiday season.",
        "He planned the itinerary carefully for the entire vacation.",
    ],
    "business": [
        "The company reported strong earnings for the quarter.",
        "The meeting was rescheduled because of a calendar conflict.",
        "The marketing team launched a new campaign this month.",
        "Investors are watching the market closely this week.",
        "The manager approved the budget for the new project.",
        "Sales increased after the product redesign was released.",
        "The startup raised funding to expand its operations.",
    ],
    "science": [
        "The experiment confirmed the original hypothesis after review.",
        "Researchers published their findings in a scientific journal.",
        "The telescope captured a clear image of the distant galaxy.",
        "The study examined the effects of sleep on memory.",
        "Scientists discovered a new species during the expedition.",
        "The laboratory results were consistent across every trial.",
    ],
    "daily_life": [
        "She walked the dog around the park every evening.",
        "He forgot his umbrella at home this morning.",
        "The kids played in the garden until it got dark.",
        "I need to buy groceries before the store closes.",
        "They cleaned the house before the guests arrived.",
        "She read a book quietly by the window all afternoon.",
        "He fixed the broken chair in the kitchen yesterday.",
        "We watched a movie together on Friday night.",
    ],
}


def build_corpus(repeat: int = 6, seed: int = 42) -> str:
    """
    Build a diverse text corpus by shuffling and repeating sentences from
    several topics. Repetition with shuffling creates realistic word-
    frequency statistics (some words/phrases recur often, like in real
    language) without literally duplicating identical paragraphs.
    """
    rng = random.Random(seed)
    all_sentences = []
    for sentences in _TOPIC_SENTENCES.values():
        all_sentences.extend(sentences)

    corpus_sentences = []
    for _ in range(repeat):
        shuffled = all_sentences.copy()
        rng.shuffle(shuffled)
        corpus_sentences.extend(shuffled)

    return " ".join(corpus_sentences)


if __name__ == "__main__":
    text = build_corpus()
    print(f"Corpus length: {len(text)} characters, "
          f"{len(text.split())} words")
