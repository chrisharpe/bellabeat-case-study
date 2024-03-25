BellaBeat Case Study Capston Project:  Fitbit Activity Analysis

Project Overview:
This repository contains a Python script for analyzing public Fitbit activity data. The script loads, processes, and visualizes data to identify and compare different physical activities, such as running, cycling, and weightlifting. Insights are drawn from hourly intensities, hourly steps, and daily activity metrics to understand activity patterns and intensity. These insights are used to help inform digital marketing campaigns for BellaBeat. 

Data Source:
The data used in this analysis is publicly available on Kaggle, provided by Fitabase.

The dataset can be accessed here: https://www.kaggle.com/datasets/arashnic/fitbit

The metadata can be accessed here: https://www.fitabase.com/media/1930/fitabasedatadictionary102320.pdf

Getting Started:

Prerequisites:
Before running the script, ensure you have Python installed on your system. This project was developed using Python 3.8. You'll also need to install the required Python packages.

Installation:
Clone this repository or download the ZIP file.

Navigate to the project directory. Save the following .csv files to the project directory from the dataset:  dailyActivity_merged.csv, hourlyIntensities_merged.csv, hourlySteps_merged.csv

Install the required packages by running:

pip install -r requirements.txt

Running the Script:
To analyze the Fitbit data and generate visualizations, run the following command in the project directory:

python3 data_analysis.py

Features:
Data Loading: Reads hourly intensities, hourly steps, and daily activities from CSV files.
Data Preparation: Merges datasets, calculates additional metrics, and filters out irrelevant data.
Activity Identification: Identifies instances of running, cycling, and weightlifting based on predefined criteria.
Visualization: Generates pie charts and bar graphs to compare the frequency of each activity type across users.

Dependencies:
The script depends on several Python libraries for data manipulation and visualization, including pandas, matplotlib, and numpy. A complete list of dependencies is provided in the requirements.txt file.

Contributing:
Contributions to enhance the functionality or improve the code are welcome. Please feel free to fork the repository and submit pull requests.