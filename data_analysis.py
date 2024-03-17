import pandas as pd
import matplotlib.pyplot as plt


def load_data():
    hourly_intensities = pd.read_csv('hourlyIntensities_merged.csv')
    hourly_intensities['ActivityHour'] = pd.to_datetime(
        hourly_intensities['ActivityHour'], errors='coerce')

    hourly_steps = pd.read_csv('hourlySteps_merged.csv')
    hourly_steps['ActivityHour'] = pd.to_datetime(
        hourly_steps['ActivityHour'], errors='coerce')

    daily_activity = pd.read_csv('dailyActivity_merged.csv')
    daily_activity['ActivityDate'] = pd.to_datetime(
        daily_activity['ActivityDate'], errors='coerce')

    return hourly_intensities, hourly_steps, daily_activity


def prepare_data(hourly_intensities, hourly_steps, daily_activity):
    # Calculate median steps for running identification
    median_steps = hourly_steps.groupby(
        'Id')['StepTotal'].median().reset_index(name='MedianStepCount')

    # Merge hourly datasets for a comprehensive view
    hourly_data = pd.merge(hourly_steps, hourly_intensities[[
                           'Id', 'ActivityHour', 'TotalIntensity']], on=['Id', 'ActivityHour'])

    # Ensure the 'Date' column for merging is just the date part, derived from 'ActivityHour'
    hourly_data['Date'] = hourly_data['ActivityHour'].dt.date

    # Calculate ActiveDistance for cycling identification
    daily_activity['TotalActiveDistance'] = daily_activity['VeryActiveDistance'] + \
        daily_activity['ModeratelyActiveDistance']

    # Convert 'ActivityDate' to just the date part for consistency
    daily_activity['ActivityDate'] = daily_activity['ActivityDate'].dt.date

    return median_steps, hourly_data, daily_activity


def identify_activities(median_steps, hourly_data, daily_activity):
    # Prepare the hourly_data DataFrame by merging necessary data
    hourly_data = pd.merge(hourly_data, median_steps, on='Id')
    hourly_data = pd.merge(hourly_data, daily_activity[['Id', 'ActivityDate', 'TotalActiveDistance']], left_on=[
                           'Id', 'Date'], right_on=['Id', 'ActivityDate'])
    # Ensure data is sorted
    hourly_data.sort_values(by=['Id', 'ActivityHour'], inplace=True)

    # Running
    hourly_data['IsRunning'] = (hourly_data['TotalActiveDistance'] > 3.2) & (
        hourly_data['StepTotal'] > hourly_data['MedianStepCount'])

    # Mark start of new running instances (when previous row is not running, but current is)
    hourly_data['PrevIsRunning'] = hourly_data.groupby('Id')[
        'IsRunning'].shift(1)
    hourly_data['StartNewRun'] = (hourly_data['IsRunning']) & (
        hourly_data['PrevIsRunning'] != True)

    # Assign an instance ID to each running block
    hourly_data['RunningInstance'] = hourly_data.groupby('Id')[
        'StartNewRun'].cumsum()

    # Cycling
    daily_activity['IsCyclingDay'] = (daily_activity['TotalActiveDistance'] > 20) & \
                                     ((daily_activity['TotalSteps'] / daily_activity['TotalActiveDistance']) <= (
                                         2050 / 1.6))
    hourly_data['IsCycling'] = hourly_data['Date'].isin(
        daily_activity[daily_activity['IsCyclingDay']]['ActivityDate'])

    # Weightlifting
    hourly_data['IsWeightlifting'] = (hourly_data['TotalIntensity'] > 50) & (
        hourly_data['StepTotal'] < 2000)

    # Collapse hourly running data into a summary of instances
    hourly_data['IsRunningInstance'] = hourly_data['StartNewRun'] & hourly_data['IsRunning']
    running_instances_summary = hourly_data.groupby(
        'Id')['IsRunningInstance'].sum().reset_index(name='RunningInstances')
    cycling_instances_summary = hourly_data.groupby(
        'Id')['IsCycling'].sum().reset_index(name='CyclingInstances')
    weightlifting_instances_summary = hourly_data.groupby(
        'Id')['IsWeightlifting'].sum().reset_index(name='WeightliftingInstances')

    # Merge summaries for visualization
    summary = running_instances_summary
    summary = summary.merge(cycling_instances_summary, on='Id')
    summary = summary.merge(weightlifting_instances_summary, on='Id')

    return summary


def predict_activities():
    hourly_intensities, hourly_steps, daily_activity = load_data()
    median_steps, hourly_data, daily_activity_prepared = prepare_data(
        hourly_intensities, hourly_steps, daily_activity)
    activity_instances = identify_activities(
        median_steps, hourly_data, daily_activity_prepared)
    return activity_instances


def visualize_activities(activity_instances):
    # Pie chart
    total_running = activity_instances['RunningInstances'].sum()
    total_cycling = activity_instances['CyclingInstances'].sum()
    total_weightlifting = activity_instances['WeightliftingInstances'].sum()

    activity_counts = {
        'Running': total_running,
        'Cycling': total_cycling,
        'Weightlifting': total_weightlifting,
    }

    plt.figure(figsize=(8, 8))
    plt.pie(activity_counts.values(), labels=activity_counts.keys(),
            autopct='%1.1f%%', startangle=140)
    plt.title('Activity Instances Comparison')
    plt.show()

    # Bar graph for each user's activity instances
    activity_instances.set_index('Id').plot(
        kind='bar', stacked=True, figsize=(14, 8))
    plt.title('Number of Activity Instances per User')
    plt.ylabel('Number of Instances')
    plt.xticks(rotation=45)
    plt.legend(['Running Instances', 'Cycling Instances',
               'Weightlifting Instances'])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    activity_instances = predict_activities()
    visualize_activities(activity_instances)
    # print(activity_instances[['ActivityHour', 'IsRunning',
    #       'IsCycling', 'IsWeightlifting']].head())
