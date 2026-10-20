# FocusBuddy

## Authors

- Author #1: Vaishnavi Panchal
- Author #2: Aradhana (Ara) Anandkumar 

## Project Description

FocusBuddy is a Python-based study buddy application intended to help students understand their study and distraction patterns and pinpoint the best studying regime for them. Our project will use a computer-vision model to determine whether a student is focused, distracted, or away from their workspace during a study session. After taking this data, it will then record session information such as session duration, focus percentage, and distraction periods. Users will then be able to fill in study subject and goal completion status. The application will then analyze this information, creating specific statistics and visualization, allowing users to then compare their current study pattens with public study datasets about habits, concentration, and productivity. The application will then suggest effective study habits based on the users current study routine and public dataset.


## Project Outline/Plan

### Interface Plan

The application will be created using Python’s Tkinter library. It will include a main window for starting and managing study sessions and a separate results window for viewing analysis and visualizations.

Our project will utilize Python's Tkinter library, and wiw consist of a main window used for study session starting and managing. A separate window will contain information for analyziz and visualization, as well as a data export option.

The main window will include:
- Study subject text input
- Study goal text input
- Duration input
- Start/pause/end session button
- Timer
- Label to determine users focus status
- Webcam status label
- button to navigate to previous sessions (taking to second window)

The results window will include:

- previous session view and drop down
- date and subject filter
- total study time label
- focus time label
- distraction time label
- focus percentage
- button to generate a visualization
- button to compare results with external data
- button to export data

The application will use the users webcam input in order to analyze a user's study vs distraction period. Specifically, the model will analyze if a users face is present, what the user is doing, and whether they are engaged in their workspace. It will then classify this information into focus, distracted, or away labels. 

### Data Collection and Storage Plan

*Written by Author #1: Vaishnavi Panchal*

The overal project will collect data through the user's study sessions, as well utilize a public web accessed dataset. When a user starts a session, the application will record the session id, data, subject, study goal, start time, and duration. The computer vision model will then check the user in intervals, in order to determine whether a user is focused or distracted based on their face detection, head direction, and focus status. 

User's will also have the option to self report their focus level, goal completion, and if they faced any distrctions. Additionally, the project will use the Python requests library in order to obtain a public dataset involving study habits, concentration, and productivity, which will then be compared with the user session. The goal of the public dataset is to provide context and comparison among broader trends. 

The collected data will be stored as follows:
```text
data/
├── study_sessions.csv
├── focus_events.csv
├── student_feedback.csv
└── external_study_data.csv
```

### Data Analysis and Visualization Plan 
_Written by Author #2: Aradhana (Ara) Anandkumar_

Our project will entail parsing through users' webcam footage of a certain study session to discern between periods of focus and distraction; and, to differentiate between such periods may vary across user, so the process of parsing through webcam footage with a CV model is vital to deriving key patterns. Our project will process corresponding focus labels (e.g. focused, distracted, afk) produced by the CV model to separate periods of focus from periods of distraction. To account for variability per user, we will smooth the raw labels and only count a distraction or away period if it lasts longer than a short threshold (e.g., 5 to 10 seconds).

We will be utilizing Pandas heavily in our project; primarily to combine `study_sessions.csv`, `focus_events.csv`, and `student_feedback.csv` to calculate the following: 

- Total study, focus, distraction, and afk (away from keyboard) time 
- Number of distractions _per_ study session
- Goal focused (represented as a completion rate) that is paired with _target subjects_ (e.g. Calculus focus sprint, Computer Science focus sprint, etc) and by _time_

The visualized data will be stored as follows: 
```
output/
├── focus_timeline.png
├── focus_by_subject.png
├── focus_over_time.png
├── comparison_with_external_data.png
└── exported_session_summary.csv
``







