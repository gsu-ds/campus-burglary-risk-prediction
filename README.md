# Data Science Capstone Project (Fall 2025)

## **Spatiotemporal Forecasting of Burglary Risk in Atlanta**

## Abstract 

Routine Activity Theory posits that crime requires the convergence of a motivated offender, a suitable target, and the lack of a capable guardian. Applied to Atlanta’s university districts, this framework highlights the urgent need for proactive rather than reactive safety measures. This project focuses on forecasting burglary and larceny risks across Atlanta’s 25 Neighborhood Planning Units (NPUs) to improve campus safety.

Leveraging the Atlanta Police Department's Open Data Portal (2021–Present), we constructed an automated ETL pipeline to engineer spatiotemporal features, including semester schedules and time-of-day dynamics. We benchmark traditional time-series models against machine learning algorithms (Random Forest, XGBoost, & Prophet) to forecast incident counts. 

Models are evaluated using RMSE and MAE to assess their utility for real-world resource allocation.

Our final deliverable is an interactive GIS dashboard allowing stakeholders to visualize predicted risk levels. By identifying daily and seasonal trends, this tool empowers university administrators, law enforcement, and students to make data-driven decisions regarding patrol staffing and safer housing choices.



## 🧑‍🚀 Team

We are an student research team bringing together our technical expertise to build models aimed at reducing burglary risk near Metro Atlanta college campuses.

Data Science Team (Alphabetically): Gunn Madan, Harini Mohan, Joshua Piña, Yuntian Wu


## Goals

- To design and implement a predictive modeling and visualization system that:
   - Forecasts hourly and daily burglary risk in each of Atlanta's 25 NPU's.
   - Delivers actionable insights through an interactive dashboard to support proactive safety strategies.


## Infrastructure & Tech Stack
- Communication Tools: [Slack](https://join.slack.com/t/gsudatascienc-2cp1426/shared_invite/zt-3e29bsar7-I0lsBoRp1i8J1o6TkleC3w)
- Version Control System: [GitHub](https://github.com/gsu-ds/campus-burglary-risk-prediction)
- Software Development/Data Storage Solution: Streamlit and Supabase(PostgreSQL)
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

👉 [Project Page (New Application Coming Soon)](https://google.com)

---
