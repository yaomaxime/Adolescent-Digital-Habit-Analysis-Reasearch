import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

# 1. LOAD DATA
# Replace 'your_data.csv' with the name of your file
# Ensure you save your Excel sheet as a CSV file first
df = pd.read_csv('repositorio_translated.csv')

# 2. DATA CLEANING (Fixing the comma-decimal issue)
# This converts columns with commas (e.g., '6,4') into numbers (6.4)
cols_to_fix = [
    'Study_Hours_per_Day', 'Sleep_Hours', 'Attendance_%', 
    'Assignments_Completed_%', 'Midterm_Score', 'Final_Score', 
    'Stress_Level_(1-10)', 'Social_Media_Hours', 'GPA'
]

for col in cols_to_fix:
    if df[col].dtype == 'object':
        df[col] = df[col].str.replace(',', '.').astype(float)

# 3. GENERATE DESCRIPTIVE STATISTICS TABLE
# Useful for your "Results" or "Methods" section
stats_table = df.describe().transpose()
stats_table.to_csv('descriptive_statistics.csv')
print("Descriptive Statistics Table Created.")

# 4. CALCULATE CORRELATIONS
# This gives you the 'r' values for all variables
correlation_matrix = df[cols_to_fix].corr()
print("\nCorrelation Coefficient (r) with GPA:")
print(correlation_matrix['GPA'].sort_values())

# 5. GRAPH 1: SCATTER PLOT (Social Media vs Sleep)
plt.figure(figsize=(8, 6))
sns.regplot(x='Social_Media_Hours', y='Sleep_Hours', data=df, 
            scatter_kws={'alpha':0.5}, line_kws={'color':'red'})

# Calculate R-squared for the graph
slope, intercept, r_value, p_value, std_err = stats.linregress(df['Social_Media_Hours'], df['Sleep_Hours'])
plt.title(f'Social Media vs Sleep\n(R² = {r_value**2:.3f}, r = {r_value:.3f})')
plt.xlabel('Daily Social Media Usage (Hours)')
plt.ylabel('Sleep Duration (Hours)')
plt.savefig('scatter_sleep_social.png')
plt.close()

# 6. GRAPH 2: SCATTER PLOT (Social Media vs Stress)
plt.figure(figsize=(8, 6))
sns.regplot(x='Social_Media_Hours', y='Stress_Level_(1-10)', data=df, 
            scatter_kws={'alpha':0.5}, line_kws={'color':'red'})

slope, intercept, r_value, p_value, std_err = stats.linregress(df['Social_Media_Hours'], df['Stress_Level_(1-10)'])
plt.title(f'Social Media vs Stress\n(R² = {r_value**2:.3f}, r = {r_value:.3f})')
plt.xlabel('Daily Social Media Usage (Hours)')
plt.ylabel('Reported Stress Level (1-10)')
plt.savefig('scatter_stress_social.png')
plt.close()

# 7. GRAPH 3: BOXPLOT (GPA by Gender)
# Excellent for identifying distribution and outliers
plt.figure(figsize=(8, 6))
sns.boxplot(x='Gender', y='GPA', data=df, palette='Set2')
plt.title('Distribution of GPA by Gender')
plt.savefig('boxplot_gpa_gender.png')
plt.close()

print("\nAll graphs and tables have been saved to your folder.")