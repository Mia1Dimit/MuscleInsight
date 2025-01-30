# Muscle Insight Project Definition


The "Muscle Insight" project is dedicated to advancing the understanding and monitoring of muscle fatigue by analyzing surface electromyography (sEMG) signals. By collecting extensive sEMG data from various individuals performing a standardized leg extension exercise, the project focuses on extracting some correlation out of meaningful metrics and muscle fatigue. This data-driven approach aims to uncover patterns and relationships in muscle activity, paving the way for innovative applications in health, sports science, and rehabilitation.


Link to Electronics Special Issue details: https://www.mdpi.com/journal/electronics/special_issues/Wireless_Sensor_Network

# Project Timeline & Tasks

## 1. Initial Setup & Validation
- [x] Define project title, Write abstract
- [x] Get approval 

## 2. Window Size Analysis
- [x] Document current window sizes and steps
- [x] Analyze each metric calculation with different window sizes
      -Window sizes -> 200, 400, 800, 1600     <br>
      -Overlap (%) -> 25%, 50%, 75%, 87.5%  <br>
- [x] Create documentation comparing window sizes
      -Results: -Best performing window -> 800 with overlap as 75% or 87.5%<br>
               -Best performing metrics -> mnf_arv_ratio, ima_diff <br>

## 3. Baseline Research
- [x] Evaluate idle
- [ ] Determine calibration needs (idle vs active)
      Check if for every person's 1st iteration these ratios are comparable.<br>
      1. rms( active / (rest rms) ) -> different range of values among different participants<br>
      2. metrics ( active / (rest rms) ) -> as a result of 1. not an insightful way<br>
      3. metrics(active) / metrics(rest) -> good for some metrics, variance in the values of mnf_arv_ratio and ima_diff<br>
      4. rms ( active / (1stActiveWindow rms) ) -> some variation among participants <br>
      5. metrics ( active / (1stActiveWindow rms) ) -> more or less stable behavior with the majority of the metrics following a standard range, with some exceptions <br>
      6. metrics(active) / metrics(1stActiveWindow)<br>


## 4. Fatigue Analysis
- [ ] Develop fatigue metric regression
- [ ] Train
- [ ] Evaluation
      
## 5. Algorithm Implementation
- [ ] Develop calibration code
- [ ] Implement complete algorithm

## 6. Arduino BLE Communication
- [ ] Define BLE data transmission logic
