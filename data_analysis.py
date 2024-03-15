import pandas as pd
import matplotlib.pyplot as plt


def load_data(file_path):
    return pd.read_csv(file_path)


def calculate_ratio(row):
    active_minutes = row['VeryActiveMinutes'] + row['FairlyActiveMinutes']
    return row['TotalDistance'] / active_minutes if active_minutes > 0 else None


def categorize_activity_with_ratio(row, median_steps, median_distance):
    ratio = calculate_ratio(row)
    if row['TotalSteps'] > median_steps and row['TotalDistance'] > median_distance:
        return 'Running'
    elif row['TotalSteps'] < median_steps and row['TotalDistance'] > median_distance:
        return 'Cycling'
    elif ratio is not None and ratio <= (0.5 / 15):
        return 'Gym'
    else:
        return 'Other'


def calculate_thresholds(activity_data):
    median_steps = activity_data['TotalSteps'].median()
    median_distance = activity_data['TotalDistance'].median()
    return median_steps, median_distance


def categorization(activity_data, median_steps, median_distance):
    activity_data['ActivityCategory'] = activity_data.apply(
        lambda row: categorize_activity_with_ratio(row, median_steps, median_distance), axis=1
    )
    return activity_data


def count_activity_days(activity_data):
    return activity_data.groupby('Id')['ActivityCategory'].value_counts().unstack(fill_value=0)


def plotting(activity_counts):
    activity_counts_filtered = activity_counts.drop(columns=['Other'])
    activity_counts_filtered.plot(kind='bar', stacked=True, figsize=(14, 8))
    plt.title('Activity Type by User')
    plt.xlabel('User ID')
    plt.ylabel('Number of Days')
    plt.legend(title='Activity')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def overall_activity_category_counts(activity_data):
    activity_counts = activity_data['ActivityCategory'].value_counts()
    return activity_counts


def plot_activity_pie_chart(activity_counts):
    activity_counts_filtered = activity_counts[[
        'Running', 'Cycling', 'Gym', 'Other']]
    plt.figure(figsize=(10, 7))
    plt.pie(activity_counts_filtered, labels=activity_counts_filtered.index,
            autopct='%1.1f%%', startangle=140)
    plt.title('Percentage of Activities: Running vs Cycling vs Gym')
    plt.show()


def main():
    file_path = 'dailyActivity_merged.csv'
    activity_data = load_data(file_path)
    median_steps, median_distance = calculate_thresholds(activity_data)
    activity_data = categorization(
        activity_data, median_steps, median_distance)
    activity_counts = count_activity_days(activity_data)
    plotting(activity_counts)
    overall_counts = overall_activity_category_counts(activity_data)
    plot_activity_pie_chart(overall_counts)


if __name__ == '__main__':
    main()
