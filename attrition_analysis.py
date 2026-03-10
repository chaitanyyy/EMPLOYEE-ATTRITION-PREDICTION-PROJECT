"""
EMPLOYEE ATTRITION PREDICTION PROJECT
Complete Analysis - Working Version
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("\n")
print("="*70)
print(" " * 15 + "EMPLOYEE ATTRITION PREDICTION PROJECT")
print("="*70)
print("\n")

# STEP 1: LOAD DATA
print("Loading Dataset...")
print("-" * 70)

import os

# Auto-detect CSV file or download
csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]

if csv_files:
    csv_filename = csv_files[0]
    print(f"Found CSV file: {csv_filename}")
    df = pd.read_csv(csv_filename)
    print("✓ Dataset loaded successfully!")
else:
    print("No CSV file found. Downloading dataset...")
    url = 'https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv'
    df = pd.read_csv(url)
    df.to_csv('employee_attrition.csv', index=False)
    print("✓ Dataset downloaded and saved!")

print(f"  • Total Employees: {len(df)}")
print(f"  • Total Features: {len(df.columns)}")
print("\nFirst 5 rows:")
print(df.head())
input("\nPress Enter to continue...")

# STEP 2: EXPLORE DATA
print("\nInitial Data Exploration...")
print("-" * 70)
print("\nAttrition Distribution:")
print(df['Attrition'].value_counts())
attrition_rate = (df['Attrition'] == 'Yes').mean() * 100
print(f"Current Attrition Rate: {attrition_rate:.2f}%")
print(f"Average Age: {df['Age'].mean():.1f} years")
print(f"Average Monthly Income: ${df['MonthlyIncome'].mean():.0f}")
input("\nPress Enter to continue...")

# STEP 3: CLEAN DATA
print("\nCleaning Data...")
print("-" * 70)
columns_to_drop = ['EmployeeNumber', 'EmployeeCount', 'StandardHours', 'Over18']
df_clean = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
df_clean['Attrition'] = df_clean['Attrition'].map({'Yes': 1, 'No': 0})
categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
df_encoded = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)
print(f"✓ Data cleaned! Shape: {df_encoded.shape}")
input("\nPress Enter to continue...")

# STEP 4: CREATE VISUALIZATIONS
print("\nCreating Visualizations...")
print("-" * 70)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Employee Attrition Analysis', fontsize=16, fontweight='bold')

# Chart 1: Attrition Count
ax1 = axes[0, 0]
stayed = (df['Attrition'] == 'No').sum()
left = (df['Attrition'] == 'Yes').sum()
ax1.bar(['Stayed', 'Left'], [stayed, left], color=['#2ecc71', '#e74c3c'])
ax1.set_title('Attrition Distribution')
ax1.set_ylabel('Count')

# Chart 2: Age Distribution
ax2 = axes[0, 1]
df[df['Attrition']=='No']['Age'].hist(ax=ax2, bins=20, alpha=0.7, label='Stayed', color='#2ecc71')
df[df['Attrition']=='Yes']['Age'].hist(ax=ax2, bins=20, alpha=0.7, label='Left', color='#e74c3c')
ax2.set_title('Age Distribution')
ax2.legend()

# Chart 3: Monthly Income
ax3 = axes[0, 2]
income_stayed = df[df['Attrition']=='No']['MonthlyIncome']
income_left = df[df['Attrition']=='Yes']['MonthlyIncome']
ax3.boxplot([income_stayed, income_left], labels=['Stayed', 'Left'])
ax3.set_title('Monthly Income')

# Chart 4: Overtime Impact
ax4 = axes[1, 0]
overtime_counts = pd.crosstab(df['OverTime'], df['Attrition'])
overtime_pct = overtime_counts.div(overtime_counts.sum(axis=1), axis=0) * 100
overtime_pct.plot(kind='bar', ax=ax4, color=['#2ecc71', '#e74c3c'], width=0.7)
ax4.set_title('Overtime Impact on Attrition')
ax4.set_xlabel('Overtime')
ax4.set_ylabel('Percentage')
ax4.set_xticklabels(['No', 'Yes'], rotation=0)
ax4.legend(['Stayed', 'Left'])

# Chart 5: Department Attrition
ax5 = axes[1, 1]
dept_counts = pd.crosstab(df['Department'], df['Attrition'])
dept_pct = dept_counts.div(dept_counts.sum(axis=1), axis=0) * 100
dept_pct.plot(kind='bar', ax=ax5, color=['#2ecc71', '#e74c3c'], width=0.7)
ax5.set_title('Department Attrition Rate')
ax5.set_xlabel('Department')
ax5.set_ylabel('Percentage')
ax5.legend(['Stayed', 'Left'])

# Chart 6: Tenure Analysis
ax6 = axes[1, 2]
tenure_attrition = df.groupby('YearsAtCompany')['Attrition'].apply(
    lambda x: (x == 'Yes').mean() * 100
)
ax6.plot(tenure_attrition.index, tenure_attrition.values, marker='o', color='#e74c3c', linewidth=2)
ax6.set_title('Attrition Rate by Years at Company')
ax6.set_xlabel('Years at Company')
ax6.set_ylabel('Attrition Rate (%)')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('eda_insights.png', dpi=300, bbox_inches='tight')
print("✓ Saved as 'eda_insights.png'")
plt.show()
input("\nPress Enter to continue...")

# STEP 5: KEY FINDINGS
print("\nKey Findings...")
print("-" * 70)
overtime_yes = df[df['OverTime']=='Yes']['Attrition'].apply(lambda x: 1 if x=='Yes' else 0).mean() * 100
overtime_no = df[df['OverTime']=='No']['Attrition'].apply(lambda x: 1 if x=='Yes' else 0).mean() * 100
print(f"Overtime workers: {overtime_yes:.1f}% attrition")
print(f"No overtime: {overtime_no:.1f}% attrition")
print(f"→ {overtime_yes/overtime_no:.1f}x higher risk!")
input("\nPress Enter to continue...")

# STEP 6: PREPARE FOR ML
print("\nPreparing for Machine Learning...")
print("-" * 70)
X = df_encoded.drop('Attrition', axis=1)
y = df_encoded['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training: {X_train.shape[0]} samples")
print(f"Test: {X_test.shape[0]} samples")
input("\nPress Enter to continue...")

# STEP 7: BUILD MODELS
print("\nBuilding Models...")
print("-" * 70)

print("\nLogistic Regression...")
log_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
y_pred_proba_log = log_model.predict_proba(X_test)[:, 1]
log_auc = roc_auc_score(y_test, y_pred_proba_log)
log_acc = (y_pred_log == y_test).mean() * 100
print(f"✓ Accuracy: {log_acc:.2f}%")
print(f"✓ ROC-AUC: {log_auc:.3f}")

print("\nRandom Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', max_depth=10)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, y_pred_proba_rf)
rf_acc = (y_pred_rf == y_test).mean() * 100
print(f"✓ Accuracy: {rf_acc:.2f}%")
print(f"✓ ROC-AUC: {rf_auc:.3f}")
input("\nPress Enter to continue...")

# STEP 8: MODEL COMPARISON
print("\nModel Performance...")
print("-" * 70)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

cm_log = confusion_matrix(y_test, y_pred_log)
cm_rf = confusion_matrix(y_test, y_pred_rf)

sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Logistic Regression')

sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('Random Forest')

fpr_log, tpr_log, _ = roc_curve(y_test, y_pred_proba_log)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
axes[2].plot(fpr_log, tpr_log, label=f'Logistic (AUC={log_auc:.3f})')
axes[2].plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={rf_auc:.3f})')
axes[2].plot([0, 1], [0, 1], 'k--')
axes[2].set_title('ROC Curves')
axes[2].legend()

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved as 'model_comparison.png'")
plt.show()
input("\nPress Enter to continue...")

# STEP 9: FEATURE IMPORTANCE
print("\nFeature Importance...")
print("-" * 70)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 Features:")
for i, row in feature_importance.head(10).iterrows():
    print(f"  {i+1}. {row['Feature']}")

plt.figure(figsize=(10, 8))
top_15 = feature_importance.head(15)
plt.barh(range(len(top_15)), top_15['Importance'].values, color='#3498db')
plt.yticks(range(len(top_15)), top_15['Feature'].values)
plt.xlabel('Importance')
plt.title('Top 15 Features Driving Attrition', fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved as 'feature_importance.png'")
plt.show()
input("\nPress Enter to continue...")

# STEP 10: AT-RISK EMPLOYEES
print("\nAt-Risk Employees...")
print("-" * 70)
all_predictions = rf_model.predict_proba(X)[:, 1]
risk_df = df.copy()
risk_df['Attrition_Risk_Score'] = (all_predictions * 100).round(1)
risk_df['Risk_Level'] = pd.cut(all_predictions, bins=[0, 0.3, 0.6, 1.0], labels=['Low', 'Medium', 'High'])

high_risk = risk_df[risk_df['Risk_Level'] == 'High'].sort_values('Attrition_Risk_Score', ascending=False)
print(f"\nHigh-Risk Employees: {len(high_risk)}")
print("\nTop 10:")
print(high_risk[['Age', 'Department', 'JobRole', 'Attrition_Risk_Score']].head(10))

risk_df.to_csv('high_risk_employees.csv', index=False)
print("\n✓ Saved as 'high_risk_employees.csv'")
input("\nPress Enter for summary...")

# FINAL SUMMARY
print("\n" + "="*70)
print(" " * 20 + "🎉 PROJECT COMPLETED! 🎉")
print("="*70)
print(f"\nBUSINESS METRICS:")
print(f"  • Attrition Rate: {attrition_rate:.2f}%")
print(f"  • High-Risk Employees: {len(high_risk)}")
print(f"\nMODEL PERFORMANCE:")
print(f"  • Best Model: Random Forest")
print(f"  • Accuracy: {rf_acc:.2f}%")
print(f"  • ROC-AUC: {rf_auc:.3f}")
print(f"\nTOP 3 DRIVERS:")
for i, row in feature_importance.head(3).iterrows():
    print(f"  {i+1}. {row['Feature']}")
print(f"\nFILES GENERATED:")
print("  ✓ eda_insights.png")
print("  ✓ model_comparison.png")
print("  ✓ feature_importance.png")
print("  ✓ high_risk_employees.csv")
print("\n" + "="*70)
print("Project completed successfully!")
print("="*70)