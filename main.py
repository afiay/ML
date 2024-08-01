import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
female_df = pd.read_csv('./data/female.csv')
male_df = pd.read_csv('./data/male.csv')

# Inspect the data
print("Female Data Shape:", female_df.shape)
print("Male Data Shape:", male_df.shape)
print(female_df.head())
print(male_df.head())

# Define the Size Charts
female_sizes = {
    'XS': {'Chest': 695, 'Shoulder': 283},
    'S': {'Chest': 824, 'Shoulder': 335},
    'M': {'Chest': 889, 'Shoulder': 353},
    'L': {'Chest': 940, 'Shoulder': 365},
    'XL': {'Chest': 999, 'Shoulder': 378},
    '2XL': {'Chest': 1057, 'Shoulder': 389},
    '3XL': {'Chest': 1117, 'Shoulder': 400}
}

male_sizes = {
    'XS': {'Chest': 774, 'Shoulder': 337},
    'S': {'Chest': 922, 'Shoulder': 384},
    'M': {'Chest': 996, 'Shoulder': 403},
    'L': {'Chest': 1056, 'Shoulder': 415},
    'XL': {'Chest': 1117, 'Shoulder': 428},
    '2XL': {'Chest': 1172, 'Shoulder': 441},
    '3XL': {'Chest': 1233, 'Shoulder': 452}
}


def create_overlapping_size_chart(original_chart):
    overlapping_chart = {}
    sizes = list(original_chart.keys())
    for i, size in enumerate(sizes):
        overlapping_chart[size] = {}
        if i == 0:
            overlapping_chart[size]['Chest'] = (
                0, original_chart[size]['Chest'])
            overlapping_chart[size]['Shoulder'] = (
                0, original_chart[size]['Shoulder'])
        elif i == len(sizes) - 1:
            overlapping_chart[size]['Chest'] = (
                original_chart[size]['Chest'] - 5, original_chart[size]['Chest'] + 1000)
            overlapping_chart[size]['Shoulder'] = (
                original_chart[size]['Shoulder'] - 5, original_chart[size]['Shoulder'] + 1000)
        else:
            overlapping_chart[size]['Chest'] = (
                original_chart[sizes[i-1]]['Chest'], original_chart[size]['Chest'] + 5)
            overlapping_chart[size]['Shoulder'] = (
                original_chart[sizes[i-1]]['Shoulder'], original_chart[size]['Shoulder'] + 5)
    return overlapping_chart


def map_size(value, size_ranges):
    for size, (low, high) in size_ranges.items():
        if low <= value <= high:
            return size
    return '3XL'


def assign_sizes(df, size_chart):
    chest_ranges = {size: chart['Chest'] for size, chart in size_chart.items()}
    shoulder_ranges = {size: chart['Shoulder']
                       for size, chart in size_chart.items()}

    df['Chest Size'] = df['chestcircumference'].apply(
        map_size, args=(chest_ranges,))
    df['Shoulder Size'] = df['biacromialbreadth'].apply(
        map_size, args=(shoulder_ranges,))
    return df


def count_matches_conflicts(df, size_chart):
    match_count = 0
    conflict_count = 0
    tie_count = 0

    for _, row in df.iterrows():
        chest_size = row['Chest Size']
        shoulder_size = row['Shoulder Size']

        if chest_size is None or shoulder_size is None:
            conflict_count += 1
            continue

        if chest_size == shoulder_size:
            match_count += 1
        elif abs(list(size_chart.keys()).index(chest_size) - list(size_chart.keys()).index(shoulder_size)) == 1:
            larger_size = max(chest_size, shoulder_size,
                              key=lambda x: list(size_chart.keys()).index(x))
            match_count += 1
            tie_count += 1
        else:
            conflict_count += 1

    return match_count, conflict_count, tie_count


# Create overlapping size charts
overlapping_female_sizes = create_overlapping_size_chart(female_sizes)
overlapping_male_sizes = create_overlapping_size_chart(male_sizes)

# Assign sizes
female_df = assign_sizes(female_df, overlapping_female_sizes)
male_df = assign_sizes(male_df, overlapping_male_sizes)

# Count matches, conflicts, and ties
female_matches, female_conflicts, female_ties = count_matches_conflicts(
    female_df, overlapping_female_sizes)
male_matches, male_conflicts, male_ties = count_matches_conflicts(
    male_df, overlapping_male_sizes)

# Print results
print(f"Female Matches: {female_matches}, Female Conflicts: {
      female_conflicts}, Female Ties: {female_ties}")
print(f"Male Matches: {male_matches}, Male Conflicts: {
      male_conflicts}, Male Ties: {male_ties}")

# Visualization
labels = ['Female Matches', 'Female Conflicts', 'Female Ties',
          'Male Matches', 'Male Conflicts', 'Male Ties']
counts = [female_matches, female_conflicts, female_ties,
          male_matches, male_conflicts, male_ties]

plt.figure(figsize=(10, 6))
plt.bar(labels, counts, color=['blue', 'orange',
        'green', 'blue', 'orange', 'green'])
plt.xlabel('Categories')
plt.ylabel('Counts')
plt.title('Number of Matches, Conflicts, and Ties for T-shirt Sizes')
plt.show()