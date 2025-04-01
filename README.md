# ACT Analysis 2022-2023: Descriptive and Comparative Statistics

## Overview
This analysis explores the performance of students who took the ACT during the 2022-2023 school year. The primary objective is to conduct descriptive statistics on various student subgroups, identify students meeting or not meeting ACT benchmarks, and analyze math sub-scores in detail. Additionally, the project seeks to identify the impact of different reporting categories on the Math composite score and analyze performance across ethnic groups. The findings will help answer critical research questions related to performance trends and problem areas across different sub-tests of the ACT.

## Goals
The main goals of this analysis are:

1. **Conduct descriptive statistics** for all students who took the ACT in the 2022-2023 school year.
2. **Conduct subgroup statistics** to identify performance trends.
3. **Identify students meeting and not meeting ACT benchmarks.**
4. **Analyze math sub-scores** and their contributions to the overall performance.
5. **Determine which reporting categories** (from the math test) have the greatest or least impact on the Math composite score.
6. **Analyze performance across different ethnicities.**
7. **Examine how math sub-scores relate to the math classes students took.**

## Research Questions
The following research questions guide the analysis:

1. What are the distributions of ACT Composite scores and ACT Super Scores (SOCS)?
2. What are the sub-scores and their distributions?
3. What percentage of students are meeting the ACT sub-score benchmarks?
4. What are the summary statistics for the eight Math reporting categories?
5. How do the reporting categories impact the overall Math score individually and collectively?
6. What is the performance across different ethnicities?
7. What are the problem areas for each subject (Math, Reading, Science, etc.)?

## Plan of Action
To address the research questions and achieve the goals, the following steps will be taken:

1. **Data Collection:**  
   Scrape data from credible sources and compile relevant information related to student performance on the ACT for the 2022-2023 school year.

2. **Data Cleaning and De-identification:**  
   Ensure that the data is cleaned and free of errors. De-identify sensitive student information to ensure privacy and compliance.

3. **Data Conversion:**  
   Convert the cleaned data into `.csv` format and import it into a Pandas DataFrame for analysis.

4. **Data Merging:**  
   Merge data from multiple sources (if applicable) into a single, unified DataFrame for analysis.

5. **Exploratory Data Analysis (EDA):**  
   Perform exploratory data analysis (EDA) to address the research questions. This will include summary statistics, distribution analysis, and subgroup analysis.

6. **Visualization:**  
   Create visualizations as needed to effectively communicate findings, including distribution charts, bar graphs, and heatmaps.

7. **Conclusions and Inferences:**  
   Draw conclusions and make inferences based on the results. Identify key trends, significant findings, and actionable insights.

## Data Structure
- The dataset is expected to include the following columns (this may vary based on the data source):
  - `Student ID`: A unique identifier for each student.
  - `ACT Composite Score`: The overall ACT score.
  - `ACT Sub-Scores`: The individual scores in Math, Reading, Science, and English.
  - `Ethnicity`: The student's ethnicity.
  - `Math Reporting Categories`: The specific categories in the Math section (e.g., Algebra, Geometry).
  - `Math Sub-Scores`: Specific sub-scores related to the Math section.
  - `Math Class`: The type of math class the student took (e.g., Algebra I, Geometry, Calculus).
  
## Expected Outcomes
By the end of the analysis, we aim to answer the research questions, identify significant trends in ACT performance, highlight areas of strength and weakness in various subject areas, and provide actionable insights that can inform educational practices and policy.

## Tools and Technologies
- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn
- **Data Formats:** CSV
- **Visualization Tools:** Matplotlib, Seaborn

## License
This analysis is intended for educational and research purposes. If you plan to use this work, ensure that proper attribution is given, and do not use it for commercial purposes without permission.
