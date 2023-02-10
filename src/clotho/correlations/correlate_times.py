"""
The functions on this script will find and count the number of times that a feature
happens at the same time in a list of dictionaries.
"""
import logging
import datetime
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def round_time(time: datetime.datetime, round_to: int = 60) -> datetime.datetime:
    """
    Round a datetime object to any time laps in seconds
    :param time: datetime object to round
    :param round_to: Closest number of seconds to round to, default 1 minute
    :return: The rounded time
    """
    try:
        seconds = (time - time.min).seconds
        rounding = (seconds + round_to / 2) // round_to * round_to
        return time + datetime.timedelta(0, rounding - seconds, -time.microsecond)
    except AttributeError:
        logger.error("Error rounding time: %s", time)
        return time


def count_pair_features(data: list[dict], id_key: str, time_key: str) -> list[dict]:
    """
    This function counts the number of times that a feature happens at the same time
    in a list of dictionaries.
    :param data: List of dictionaries with the data to compare
    :param id_key: Key for the id of the entity
    :param time_key: Key for the time of the entity
    :return: List of dictionaries with the counts for each pair of entities
    """
    # Create a dictionary to store the counts for each pair of entities
    pair_counts = defaultdict(int)
    # Create a dictionary to store the times for each entity
    entity_times = defaultdict(list)
    # Iterate through the input data and store the times for each entity
    for d in data:
        entity_id = d[id_key]
        time = d[time_key]
        entity_times[entity_id].append(time)
    # Iterate through the entities and their times
    for entity1, times1 in entity_times.items():
        for entity2, times2 in entity_times.items():
            if entity1 != entity2:
                # Compare the times for each pair of entities
                for time1 in times1:
                    for time2 in times2:
                        if time1 == time2:
                            pair_counts[frozenset([entity1, entity2])] += 1
    # Create a list to store the output data
    result = []
    # Iterate through the pair counts and create the output data
    for idx, (pair, count) in enumerate(pair_counts.items()):
        entities = list(pair)
        feature1 = entities[0]
        feature2 = entities[1]
        weight_time = count
        result.append(
            {"start": feature1, "end": feature2, "weight_time": weight_time / 2}
        )
        # One logger for 1000 pairs
        if idx % 1000 == 0:
            logger.info("Pair %s of %s", idx, len(pair_counts))
    for d in result:
        d["weight_time"] = normalize_weight(d["weight_time"])

    return result


def normalize_weight(weight: float) -> float:
    """
    Normalize the weight between 0 and 5
    :param weight: Weight to normalize
    :return: The normalized weight
    """
    max_weight = 5
    min_weight = 0
    return (weight - min_weight) / (max_weight - min_weight)


def round_and_correlate(
    data: list[dict], id_key: str, time_key: str, round_to: int = 60
) -> list[dict]:
    """
    This function rounds the time to the nearest minute and then counts the number of times
    that a feature happens at the same time in a list of dictionaries.
    :param data: List of dictionaries with the data to compare
    :param id_key: Key for the id of the entity
    :param time_key: Key for the time of the entity
    :param round_to: Closest number of seconds to round to, default 1 minute
    :return: List of dictionaries with the counts for each pair of entities
    """
    # Round the time to the nearest minute
    for item in data:
        item[time_key] = round_time(item[time_key], round_to)
    # Count the number of times that a feature happens at the same time
    same_time_count = count_pair_features(data, id_key, time_key)
    return same_time_count


if __name__ == "__main__":

    # Create a dictionary of data
    data = [
        {
            "entity_id": "1",
            "time": datetime.datetime(2023, 1, 20, 21, 37),
        },  # 21:37
        {
            "entity_id": "2",
            "time": datetime.datetime(2023, 1, 20, 21, 37),
        },  # 21:37
        {
            "entity_id": "1",
            "time": datetime.datetime(2023, 1, 20, 21, 39),
        },  # 21:39
        {
            "entity_id": "2",
            "time": datetime.datetime(2023, 1, 20, 21, 38),
        },  # 21:38
        {
            "entity_id": "1",
            "time": datetime.datetime(2023, 1, 20, 22, 39),
        },  # 22:39
        {
            "entity_id": "2",
            "time": datetime.datetime(2023, 1, 20, 22, 39),
        },  # 22:39
        {
            "entity_id": "1",
            "time": datetime.datetime(2023, 1, 20, 21, 38),
        },  # 21:38
        {
            "entity_id": "2",
            "time": datetime.datetime(2023, 1, 20, 21, 56),
        },  # 21:56
        {
            "entity_id": "3",
            "time": datetime.datetime(2023, 1, 20, 21, 36),
        },  # 21:36
        {
            "entity_id": "4",
            "time": datetime.datetime(2023, 1, 20, 21, 36),
        },  # 21:36
    ]

    # Count the number of times that a feature happens at the same time
    same_time_count = round_and_correlate(data, "entity_id", "time", 60)
