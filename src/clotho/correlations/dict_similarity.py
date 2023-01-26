""""
This module calculates the similarity between the languages of the channels
"""
from clotho.constants import WEIGHT_PREFIX


def measure_similarity(list_of_dicts: list[dict]) -> list[dict]:
    """
    Calculates the similarity between the languages of the channels
    :param list_of_dicts: List of dictionaries with the languages of the channels
    :return: List of dictionaries with the similarity between the languages of the channels
    """
    similarity_list = []
    for i, chat1 in enumerate(list_of_dicts):
        for j in range(i + 1, len(list_of_dicts)):
            chat2 = list_of_dicts[j]
            chat_id1 = chat1["entity_id"]
            chat_id2 = chat2["entity_id"]
            language1 = chat1["language"]
            language2 = chat2["language"]
            similarity = calculate_similarity(language1, language2)
            similarity_list.append(
                {"start": chat_id1, "end": chat_id2, f"{WEIGHT_PREFIX}lang": similarity}
            )
    return similarity_list


def calculate_similarity(language1: dict[str, int], language2: dict[str, int]) -> float:
    """
    Calculates the similarity between the languages of the channels
    :param language1: Languages of the first channel
    :param language2: Languages of the second channel
    :return: Similarity between the languages of the channels
    """
    set1 = set(language1.keys())
    set2 = set(language2.keys())
    # TODO Check how to include the count on the correlation
    try:
        jaccard = len(set1.intersection(set2)) / len(set1.union(set2))
        return jaccard
    except ZeroDivisionError:
        return 0.0


if __name__ == "__main__":

    test_dict = [
        {"entity_id": "-1001495281876", "language": {"no": 12, "en": 149}},
        {"entity_id": "-1001622654998", "language": {"en": 532, "no": 1}},
        {
            "entity_id": "-1001501522090",
            "language": {"en": 22, "Too short": 16, "fr": 1},
        },
    ]
    from pprint import pprint

    pprint(measure_similarity(test_dict))
