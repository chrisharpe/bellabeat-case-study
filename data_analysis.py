import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_data():
    hourly_intensities = pd.read_csv('hourlyIntensities_merged.csv')
    hourly_intensities['ActivityHour'] = pd.to_datetime(
        hourly_intensities['ActivityHour'], format="%m/%d/%Y %I:%M:%S %p", errors='coerce')

    hourly_steps = pd.read_csv('hourlySteps_merged.csv')
    hourly_steps['ActivityHour'] = pd.to_datetime(
        hourly_steps['ActivityHour'], format="%m/%d/%Y %I:%M:%S %p", errors='coerce')

    daily_activity = pd.read_csv('dailyActivity_merged.csv')
    daily_activity['ActivityDate'] = pd.to_datetime(
        daily_activity['ActivityDate'], format="%m/%d/%y", errors='coerce')

    return hourly_intensities, hourly_steps, daily_activity


def prepare_data(hourly_intensities, hourly_steps, daily_activity):
    # Merge hourly datasets for a comprehensive view
    hourly_data = pd.merge(hourly_steps, hourly_intensities[[
                           'Id', 'ActivityHour', 'TotalIntensity']], on=['Id', 'ActivityHour'])

    # Format Data column
    hourly_data['Date'] = hourly_data['ActivityHour'].dt.date

    # Remove entries with zero steps
    daily_activity = daily_activity[daily_activity['TotalSteps'] > 100]

    # Calculate ActiveDistance for cycling identification
    daily_activity.loc[:, 'TotalActiveDistance'] = daily_activity['VeryActiveDistance'] + \
        daily_activity['ModeratelyActiveDistance']

    # Convert 'ActivityDate' to date
    daily_activity['ActivityDate'] = daily_activity['ActivityDate'].dt.date

    # Calculate the AMStepRatio
    daily_activity.loc[:, 'AMStepRatio'] = (
        daily_activity['VeryActiveMinutes'] + daily_activity['FairlyActiveMinutes']) / daily_activity['TotalSteps']

    # Handle potential division by zero issues by replacing inf with NaN
    daily_activity.loc[:, 'AMStepRatio'] = daily_activity['AMStepRatio'].replace([
                                                                                 np.inf, -np.inf], np.nan)

    # Drop rows where AMStepRatio is NaN (cases where TotalSteps is zero)
    daily_activity.dropna(subset=['AMStepRatio'], inplace=True)

    return hourly_data, daily_activity


def identify_activities(hourly_data, daily_activity):
    # Prepare the hourly_data DataFrame by merging necessary data
    merged_data = pd.merge(hourly_data, daily_activity[['Id', 'ActivityDate', 'TotalActiveDistance', 'AMStepRatio']], left_on=[
                           'Id', 'Date'], right_on=['Id', 'ActivityDate'])
    # Ensure data is sorted
    merged_data.sort_values(by=['Id', 'ActivityHour'], inplace=True)

# Running
    merged_data['IsRunning'] = (merged_data['TotalActiveDistance'] > 3.2) & \
                               (merged_data['TotalIntensity'] > 40) & \
                               (merged_data['StepTotal'] > 2500)

    # Mark start of new running instances (when previous row is not running, but current is)
    merged_data['PrevIsRunning'] = merged_data.groupby('Id')[
        'IsRunning'].shift(1)
    merged_data['StartNewRun'] = (merged_data['IsRunning']) & \
                                 (merged_data['PrevIsRunning'] != True)

    # Assign an instance ID to each running block
    merged_data['RunningInstance'] = merged_data.groupby('Id')[
        'StartNewRun'].cumsum()


# Cycling
    # Calculate the user-specific 75th percentile of AMStepRatio
    user_specific_75th = daily_activity.groupby('Id')['AMStepRatio'].quantile(
        0.75).reset_index(name='User75thPercentile')

    # Merge this user-specific percentile back into the daily_activity dataframe
    daily_activity = pd.merge(daily_activity, user_specific_75th, on='Id')

    # Calculate ActiveDistance for each day
    daily_activity['ActiveDistance'] = daily_activity['VeryActiveDistance'] + \
        daily_activity['ModeratelyActiveDistance'] + \
        daily_activity['LightActiveDistance']

    # Refine cycling detection with the ActiveDistance criteria and exclude days with zero steps
    daily_activity['IsCyclingDay'] = (daily_activity['AMStepRatio'] > daily_activity['User75thPercentile']) & (
        daily_activity['TotalActiveDistance'] > 20)

# Weightlifting
    merged_data['IsWeightlifting'] = (merged_data['TotalIntensity'] > 50) & (
        merged_data['StepTotal'] < 2000)

    # Mark start of new weightlifting sessions
    merged_data['PrevIsWeightlifting'] = merged_data.groupby(
        'Id')['IsWeightlifting'].shift(1)
    merged_data['StartNewWeightliftingSession'] = (
        merged_data['IsWeightlifting']) & (merged_data['PrevIsWeightlifting'] != True)

    # Assign an instance ID to each weightlifting block
    merged_data['WeightliftingInstance'] = merged_data.groupby(
        'Id')['StartNewWeightliftingSession'].cumsum()

    # Collapse data into a summary of instances
    merged_data['IsRunningInstance'] = merged_data['StartNewRun'] & merged_data['IsRunning']
    running_instances_summary = merged_data.groupby(
        'Id')['IsRunningInstance'].sum().reset_index(name='RunningInstances')
    cycling_instances_summary = daily_activity.groupby(
        'Id')['IsCyclingDay'].sum().reset_index(name='CyclingInstances')
    weightlifting_instances_summary = merged_data[merged_data['IsWeightlifting']].groupby(
        'Id')['WeightliftingInstance'].nunique().reset_index(name='WeightliftingInstances')
    # Merge summaries for visualization
    summary = pd.merge(cycling_instances_summary,
                       running_instances_summary, on='Id', how='outer')
    summary = pd.merge(
        summary, weightlifting_instances_summary, on='Id', how='outer')

    # Replace NaNs with 0 for activities without instances
    summary.fillna(0, inplace=True)

    return summary


def categorize_users(summary):
    def user_category(row):
        categories = ['Cycling', 'Running', 'Weightlifting']
        counts = [row['CyclingInstances'],
                  row['RunningInstances'], row['WeightliftingInstances']]
        active_categories = [category for category,
                             count in zip(categories, counts) if count >= 7]

        if len(active_categories) > 1:
            return 'Cross-Trainer'
        elif len(active_categories) == 1:
            return active_categories[0]
        else:
            return 'Inactive'

    summary['UserCategory'] = summary.apply(user_category, axis=1)
    return summary


def predict_activities():
    hourly_intensities, hourly_steps, daily_activity = load_data()
    hourly_prepared, daily_prepared = prepare_data(
        hourly_intensities, hourly_steps, daily_activity)
    activity_instances = identify_activities(
        hourly_prepared, daily_prepared)
    return activity_instances


def visualize_activities(activity_instances, user_categories):
    # Define colors for each activity
    activity_colors = {
        'Running': '#57A26E',
        'Cycling': '#E47979',
        'Weightlifting': '#0097B2',
    }

    # Pie chart
    total_running = activity_instances['RunningInstances'].sum()
    total_cycling = activity_instances['CyclingInstances'].sum()
    total_weightlifting = activity_instances['WeightliftingInstances'].sum()

    activity_counts = {
        'Running': total_running,
        'Cycling': total_cycling,
        'Weightlifting': total_weightlifting,
    }

    # Extract colors for the pie chart based on the activity keys
    pie_colors = [activity_colors[activity]
                  for activity in activity_counts.keys()]

    plt.figure(figsize=(8, 8))
    plt.pie(activity_counts.values(), labels=activity_counts.keys(), colors=pie_colors,
            autopct='%1.1f%%', startangle=140)
    plt.title('Activity Instances Comparison')
    plt.show()

   # Bar graph for each user's activity instances
    activity_df = activity_instances.set_index('Id')
    # Use the predefined colors for consistency
    bar_colors = [activity_colors['Cycling'],
                  activity_colors['Running'], activity_colors['Weightlifting']]

    ax = activity_df[['CyclingInstances', 'RunningInstances', 'WeightliftingInstances']].plot(
        kind='bar', stacked=True, figsize=(14, 8), color=bar_colors)
    plt.title('Number of Activity Instances per User')
    plt.ylabel('Number of Instances')
    plt.xticks(rotation=45)

    # Correcting the alignment of x-axis labels
    ax.set_xticklabels(activity_df.index, rotation=45, ha="right")

    plt.legend(['Cycling Instances', 'Running Instances',
               'Weightlifting Instances'])
    plt.tight_layout()
    plt.show()

    # 2nd pie chart for user categories
    user_categories_counts = user_categories['UserCategory'].value_counts()

    # Define colors for each category, aligning with the existing color scheme
    category_colors = {
        'Running': '#57A26E',       # Green
        'Cycling': '#E47979',       # Red
        'Weightlifting': '#0097B2',  # Blue
        'Cross-Trainer': '#F0DB4F',  # Yellow
        'Inactive': '#D3D3D3'       # Grey
    }

    # Extract colors for the pie chart based on the category index
    pie_colors = [category_colors[category]
                  for category in user_categories_counts.index]

    plt.figure(figsize=(8, 8))
    plt.pie(user_categories_counts, labels=user_categories_counts.index, colors=pie_colors,
            autopct='%1.1f%%', startangle=140)
    plt.title('User Category Distribution')
    plt.show()


if __name__ == "__main__":
    activity_instances = predict_activities()
    user_categories = categorize_users(activity_instances)
    visualize_activities(activity_instances, user_categories)
