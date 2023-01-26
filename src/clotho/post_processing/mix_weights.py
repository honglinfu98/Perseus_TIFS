import csv
from os import path
from clotho.constants import DATA_PATH, WEIGHT_PREFIX


# merge multiple lists of dictionaries -- wrapped in a list -- into one list of dictionaries based on keys start and end, wrap all the weight_* keys and values in a new dictionary called weights
def merge_lists(input: list[list[dict]]) -> list[dict]:
    """
    Merges multiple lists of dictionaries into one list of dictionaries.
    """
    # create a new dictionary with start and end as keys and a new dictionary as value
    output = {}
    for edge in input:
        for item in edge:
            # if the start and end keys are not in the output dictionary, add them
            if item["start"] not in output.keys():
                output[item["start"]] = {}
            if item["end"] not in output[item["start"]].keys():
                output[item["start"]][item["end"]] = {}
            # if the start and end keys are in the output dictionary, add the weight_* keys and values to the new dictionary
            for key, value in item.items():
                if key.startswith(WEIGHT_PREFIX):
                    output[item["start"]][item["end"]][key[7:]] = value
    # convert the output dictionary to a list of dictionaries
    output = [
        {"start": start, "end": end, "weights": weights}
        for start, end in output.items()
        for end, weights in end.items()
    ]
    return output


def mix_weights(edges: list[dict]) -> list[dict]:
    """
    Mixes the weights of the edges and adds them to the edge as a new key.
    """
    for edge in edges:
        edge["mixed_weights"] = sum(edge["weights"].values())
    return edges


# convert a list of dictionaries with certain keys and fields to csv
def create_csv(data: list[dict], file_name: str) -> None:
    """
    Creates a csv file from a list of dictionaries.
    """
    # keep only start, end and mixed_weights
    data = [
        {
            key: value
            for key, value in edge.items()
            if key in ["start", "end", "mixed_weights"]
        }
        for edge in data
    ]
    with open(path.join(DATA_PATH, file_name), "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)


def run_mix_weights(list_to_mix: list[list[dict]], file_name: str) -> list[dict]:
    """
    Runs all the functions in this file.
    """
    merged_list_of_dicts = merge_lists(list_to_mix)
    data = mix_weights(merged_list_of_dicts)
    create_csv(data, file_name)
    return data


if __name__ == "__main__":

    input_time = [
        {"start": "node1", "end": "node2", "weight_time": 10},
        {"start": "node2", "end": "node3", "weight_time": 10},
    ]

    input_lenguage = [
        {"start": "node7", "end": "node2", "weight_language": 6},
        {"start": "node1", "end": "node2", "weight_language": 6},
    ]

    input_lenght = [{"start": "node4", "end": "node3", "weight_length": 5}]

    from pprint import pprint

    pprint(run_mix_weights([input_time, input_lenguage, input_lenght], "test.csv"))
