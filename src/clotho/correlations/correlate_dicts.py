""""
This module calculates the similarity between the languages of the channels
"""
from clotho.constants import WEIGHT_PREFIX


def measure_similarity(
    list_of_dicts: list[dict], key_to_compare: str, key_node: str
) -> list[dict]:
    """
    Calculates the similarity between the languages of the channels
    :param list_of_dicts: List of dictionaries with the languages of the channels
    :return: List of dictionaries with the similarity between the languages of the channels
    """
    similarity_list = []
    for i, chat1 in enumerate(list_of_dicts):
        for j in range(i + 1, len(list_of_dicts)):
            chat2 = list_of_dicts[j]
            id_1 = chat1[key_node]
            id_2 = chat2[key_node]
            key_to_compare_1 = chat1[key_to_compare]
            key_to_compare_2 = chat2[key_to_compare]
            similarity = calculate_similarity(key_to_compare_1, key_to_compare_2)
            similarity_list.append(
                {"start": id_1, "end": id_2, f"{WEIGHT_PREFIX}lang": similarity}
            )
    return similarity_list


# def measure_similarity_all_dict(list_of_dicts: list[dict], key_node: str) -> list[dict]:


def calculate_similarity(
    key_to_compare_1: dict[str, int], key_to_compare_2: dict[str, int]
) -> float:
    """
    Calculates the similarity between the languages of the channels
    :param language1: Languages of the first channel
    :param language2: Languages of the second channel
    :return: Similarity between the languages of the channels
    """
    set1 = set(key_to_compare_1.keys())
    set2 = set(key_to_compare_2.keys())
    # TODO Check how to include the count on the correlation
    try:
        jaccard = len(set1.intersection(set2)) / len(set1.union(set2))
        return jaccard
    except ZeroDivisionError:
        return 0.0


if __name__ == "__main__":

    test_dict = [
        {
            "entity_id": "-1001495281876",
            "language": {"no": 12, "en": 149},
            "name": ["test", "test2"],
        },
        {
            "entity_id": "-1001622654998",
            "language": {"en": 532, "no": 1},
            "name": ["test", "test2"],
        },
        {
            "entity_id": "-1001501522090",
            "language": {"en": 22, "Too short": 16, "fr": 1, "no": 1},
            "name": ["test1", "test2"],
        },
        {"entity_id": "-1001501522090", "language": {}},
    ]
    from pprint import pprint

    pprint(measure_similarity(test_dict, "language", "entity_id"))
