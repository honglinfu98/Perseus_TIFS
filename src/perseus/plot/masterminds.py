from os import path
from perseus.dataset.preprocess.groudtruth_labeling import read_labeling_csv_back_to_dict
import matplotlib.pyplot as plt
from collections import Counter

from perseus.settings import PROJECT_ROOT


a = read_labeling_csv_back_to_dict("train")
b = read_labeling_csv_back_to_dict("test")
c = read_labeling_csv_back_to_dict("valid")


def process_data(data):
    counts = [sum(values.values()) for _, values in data.items()]
    frequency_counts = Counter(counts)
    frequencies = list(frequency_counts.keys())
    frequency_values = list(frequency_counts.values())
    return frequencies, frequency_values

def plot_histogram(data_a, data_b, data_c, font_size=10, title_size=12, legend_size=10, tick_label_size=10):
    freqs_a, freq_vals_a = process_data(data_a)
    freqs_b, freq_vals_b = process_data(data_b)
    freqs_c, freq_vals_c = process_data(data_c)

    all_freqs = sorted(set(freqs_a + freqs_b + freqs_c))
    freq_vals_a_dict = dict(zip(freqs_a, freq_vals_a))
    freq_vals_b_dict = dict(zip(freqs_b, freq_vals_b))
    freq_vals_c_dict = dict(zip(freqs_c, freq_vals_c))

    position_adjustment = 0.25
    positions_a = [x - position_adjustment for x in range(len(all_freqs))]
    positions_b = [x for x in range(len(all_freqs))]
    positions_c = [x + position_adjustment for x in range(len(all_freqs))]

    values_a = [freq_vals_a_dict.get(freq, 0) for freq in all_freqs]
    values_b = [freq_vals_b_dict.get(freq, 0) for freq in all_freqs]
    values_c = [freq_vals_c_dict.get(freq, 0) for freq in all_freqs]

    plt.figure(figsize=(12, 8))
    plt.bar(positions_a, values_a, color='skyblue', width=0.25, label='April 13, 2018, to January 9, 2024', align='center')
    plt.bar(positions_b, values_b, color='salmon', width=0.25, label='January 10, 2024 to February 4, 2024', align='center')
    plt.bar(positions_c, values_c, color='lightgreen', width=0.25, label='February 5, 2024 to February 16, 2024', align='center')

    plt.xlabel('Number of Masterminds', fontsize=font_size)
    plt.ylabel('Number of Cryptocurrencies', fontsize=font_size)
    # plt.title('Number of Masterminds in Cryptocurrencies Crowd Pumps Across Three Periods', fontsize=title_size)
    plt.xticks(range(len(all_freqs)), all_freqs, fontsize=tick_label_size)
    plt.yticks(fontsize=tick_label_size)
    plt.legend(fontsize=legend_size)


    plt.savefig(path.join(PROJECT_ROOT, "data", 'no_masterminds.pdf'))
    plt.show()

# Example usage:
plot_histogram(a, b, c, font_size=20, title_size=20, legend_size=20, tick_label_size=20)
