# Muscle Insight Project Definition

## Introduction

The "Muscle Insight" project is dedicated to advancing the understanding and monitoring of muscle fatigue by analyzing surface electromyography (sEMG) signals. By collecting extensive sEMG data from various individuals performing a standardized leg extension exercise, the project focuses on extracting some correlation out of meaningful metrics and muscle fatigue. This data-driven approach aims to uncover patterns and relationships in muscle activity, paving the way for innovative applications in health, sports science, and rehabilitation.


Link to Electronics Special Issue details: https://www.mdpi.com/journal/electronics/special_issues/Wireless_Sensor_Network

# Project Timeline & Tasks

## 1. Initial Setup & Validation
- [x] Define project title, Write abstract
- [x] Get approval 

## 2. Window Size Analysis
- [x] Document current window sizes and steps
- [x] Analyze each metric calculation with different window sizes
- [x] Create documentation comparing window sizes

## 3. Baseline Research
- [ ] Evaluate idle
- [ ] Determine calibration needs (idle vs active):
        - Check if for every person's 1st iteration these ratios are comparable
            1. rms( active / (rest rms) )
            2. metrics ( active / (rest rms) )
            3. metrics(active) / metrics(rest)
            4. rms ( active / (1stActiveWindow rms) )
            5. metrics ( active / (1stActiveWindow rms) )
            6. metrics(active) / metrics(1stActiveWindow)


## 4. Fatigue Analysis
- [ ] Develop fatigue metric regression
- [ ] Train
- [ ] Evaluation
      
## 5. Algorithm Implementation
- [ ] Develop calibration code
- [ ] Implement complete algorithm

## 6. Arduino BLE Communication
- [ ] Define BLE data transmission logic


## Paper Title:

Developing a Novel Muscle Fatigue Index for Wireless sEMG Sensors: Metrics and Regression Models for Real-Time Monitoring


## Paper Abstract:

Muscle fatigue impacts performance in sports, rehabilitation, and daily activities, with surface electromyography (sEMG) widely used for monitoring. In this study, we analyzed sEMG signals, evaluating time, frequency, and combined-domain metrics to identify reliable fatigue indicators. Using these metrics, we developed a novel fatigue index through regression modeling, capturing fatigue progression and enabling personalized muscle-specific assessment. Integrated into a wireless BLE-enabled sensor platform, the system combines seamless body placement, mobility, and real-time data transmission. An initial calibration phase ensures adaptation to individual muscle profiles, enhancing accuracy. By balancing on-device processing with efficient wireless communication, this platform delivers scalable, real-time fatigue monitoring across diverse applications.