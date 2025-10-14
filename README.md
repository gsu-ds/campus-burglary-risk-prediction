# Data Science Capstone Project (Fall 2025)

**Predictive Analysis for Campus Safety: Modeling Burglary Risk at Atlanta’s Major Universities**

## Abstract 

Routine Activity Theory teaches that “Crime requires a motivated offender, a suitable target, and the opportunity”. Per the 2024 U.S Census, the Atlanta Metropolitan area is the 8th largest Metropolitan area in the United States. This project focuses on analyzing burglaries and burglary related crime in areas around major college campuses in Atlanta, with the goal of forecasting risk levels and hotspots that are safety concerns for students. This study will analyze burglaries, and related crimes, within a 1-mile radius of 4 major campuses, Georgia State University, Georgia Tech, Clark Atlanta University, and Spelman College.  We will utilize the Atlanta Police Department’s (APD) Open Data Portal to access offense type and the time and coordinates of each offense. To ensure specificity and relevance, we will use spatial filters to only include
incidents that occur within a 1-mile radius of each campus, where we predict that students will be disproportionately impacted.


## 🧑‍🚀 Team

We are an student research team bringing together our technical expertise to build models aimed at reduucing burglary risk near Metro Atlanta college campuses.

### Joshua Piña

Data Science Senior graduating December 2025 with plans to pursue MS in Analytics.  
Background in Python, applied probability & statistics, data analytics, big data programming, and ML.
Experienced Program Manager in GovCon and former U.S. Army Combat Medic.<br><br>
[![GitHub](https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=white)](https://www.github.com/joshuadpina)
[![GitHub Pages](https://img.shields.io/badge/Josh's-Portfolio-blue)](https://joshuapina.github.io/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/joshuadpina/)





### Robin Wu

Data Science student and graduating this December.
Programming Software: Python, (NumPy, Pandas, Seaborn), C, JavaScript, SQL, R, VBA
Data Analytics: Tableau, Power BI, Qlik Sense, Qlik View, Google Analytics, Excel, Hadoop


### Harini Mohan

Data Science student graduating Spring 2025 with a strong foundation in statistics and data analytics. 
Passionate about using data to drive strategic business decisions.
Currently a Business Analyst at Deloitte, where I help design a platform that streamlines workflows and tool adoption across teams. 

---

### Gunn Madan

A Senior, Data Science major at Georgia State University passionate about using data-driven insights to advance technology, safety, and social impact initiatives. Currently a Break Through Tech AI Fellow partnered with Cambio Labs, where I explore applied machine learning and innovation in real-world contexts. My technical experience spans Python, SQL, R, Java, JavaScript, HTML, CSS, Bash, and Swift.
[![GitHub](https://img.shields.io/badge/GitHub-black?logo=github&logoColor=white)](https://github.com/gunnmadan)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/gunnmadan)


---

## Goals

- To design and implement a predictive modeling and visualization system that:
   - Forecasts weekly burglary risk around major Atlanta campuses.
   - Identifies spatial crime hotspots within a one-mile radius of each campus.
   - Delivers actionable insights through an interactive dashboard to support proactive safety strategies.


## Infrastructure & Tech Stack
- Communication Tools: [Slack](https://join.slack.com/t/gsudatascienc-2cp1426/shared_invite/zt-3e29bsar7-I0lsBoRp1i8J1o6TkleC3w)
- Version Control System: [GitHub](https://github.com/gsu-ds/campus-burglary-risk-prediction)
- Software Development/Data Storage Solution: Frontend: Streamlit, Backend: FastAPI, DB: PostgreSQL
- Project Management Tools: [GitHub Projects](https://github.com/orgs/gsu-ds/projects/1) + [Notion](https://www.notion.so/Quick-Links-and-Overview-Capstone-2025-Burglary-Risk-Prediction-27f054e466be80b18b73ec862545c5ed?source=copy_link)
- Document Sharing: [Google Drive](https://drive.google.com/drive/folders/1dYm1BG9t2Ah-jAVDn6VQCJ11P3_9P-fS?usp=drive_link)
- Experiment Tracking: [W&B](https://wandb.ai/joshuadariuspina)

##  Development Environment

This project uses [GitHub Codespaces](https://github.com/features/codespaces) to ensure a consistent, reproducible development setup.

--- 

### Quick Start

1. **Open in Codespaces**  
   Click the green **Code** button on this repository, then choose **Open with Codespaces** → **New codespace**.

2. **Automatic setup**  
   The dev container will automatically install Python and all required packages listed in `requirements.txt`.

3. **Activate the environment**  
   When your Codespace starts, you’re ready to run scripts and notebooks immediately.
   - If (dscvenv) does not show in terminal, follow these steps:
      - Activate virtual env: (bash)-> source dscvenv/bin/activate or powershell-> ( dscvenv\Activate\scripts)
      - Use requirements.txt to ensure installations: pip install -r requirements.txt

---

## Project Website

👉 [Project Page](https://gsu-ds.github.io/)

---
