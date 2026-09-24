\# Cricket Score Prediction and Player Classification using Machine Learning



\## Overview



This project develops a machine learning framework for cricket analytics using ball-by-ball ODI match data from Cricsheet.



The system addresses two related machine learning tasks:



1\. Cricket innings final-score prediction

2\. Cricket player role classification



The score prediction component uses match-state and temporal features to estimate the eventual final score of an innings. The player classification component categorizes players into four functional roles based on their batting and bowling statistics.



\---



\## Project Objectives



The main objectives of this project are:



\- Develop machine learning models for ODI cricket score prediction.

\- Engineer temporal features that capture recent innings behavior.

\- Compare multiple regression algorithms.

\- Study the contribution of temporal feature groups using ablation analysis.

\- Perform detailed prediction-error analysis.

\- Investigate model robustness for unusually short innings.

\- Classify players into functional cricket roles.

\- Compare multiple classification algorithms.

\- Validate player classification using 5-fold stratified cross-validation.

\- Provide a reproducible machine learning pipeline.



\---



\## Dataset



The project uses ball-by-ball One Day International (ODI) cricket data obtained from Cricsheet.



The raw match data is transformed into delivery-level records containing information such as:



\- Match ID

\- Teams

\- Venue

\- Innings

\- Over

\- Ball

\- Batter

\- Non-striker

\- Bowler

\- Runs scored

\- Extras

\- Wickets

\- Current score

\- Final innings score



The processed delivery-level dataset is stored as:



```text

data/processed/odi\_deliveries.csv

