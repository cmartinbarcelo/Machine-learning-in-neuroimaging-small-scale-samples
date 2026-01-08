# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 19:26:44 2025

@author: crist
"""

import os
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
import warnings
import numpy as np

def variables_summary(df,variables):
    grouped = df.groupby("Group")[variables].agg(["mean", "std", "var", "median"])
    summary = pd.DataFrame(index=variables)   # make variables the index

    stats = ["mean", "std", "var", "median"]
    for stat in stats:
        stat_df = grouped.xs(stat, axis=1, level=1)
        stat_df = stat_df.T
        stat_df.columns = [f"{grp} {stat}" for grp in stat_df.columns]
        summary = pd.concat([summary, stat_df], axis=1)
    return summary

def plot_demographic_variables(df, variables, o_file):
    fig, axes = plt.subplots(1, len(variables), figsize=(8, 4)) 
    axes = axes.flatten()

    for i, var in enumerate(variables[:len(axes)]):
        ax = axes[i]
        if var == "sex" or var=="Sex":
            # sex
            sns.countplot(df, x="Group", hue=var, palette="coolwarm", ax=ax)
            
        else:   # if variable is numeric
            sns.boxplot(df, x="Group", y=var, palette="coolwarm", ax=ax)
    plt.savefig(o_file)
    plt.show()
    return

def density_plot_variables(df, variables, o_file):
    fig, axes = plt.subplots(6, 2, figsize=(16, 25)) 
    axes = axes.flatten()

    for i, var in enumerate(variables[:len(axes)]):
        ax = axes[i]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sns.kdeplot(
               data=df, x=var, hue="Group",
               fill=True, common_norm=False,
               palette="coolwarm", ax=ax, alpha=0.6
           )
        ax.set_title(var, fontsize=18)
        ax.set_xlabel("")
        ax.set_ylabel("Density", fontsize=16)

        ax.tick_params(axis='both', labelsize=12)
    plt.savefig(o_file)
    plt.show()
    return

def boxplots_variables(df, variables, o_file):
    fig, axes = plt.subplots(6, 2, figsize=(16, 25))
    axes = axes.flatten()

    for i, var in enumerate(variables[:len(axes)]):
        ax = axes[i]

        # Skip variables that have no numeric data
        if df[var].dropna().empty:
            ax.set_title(f"{var} (no data)")
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sns.boxplot(
                data=df, x="Group", y=var,
                palette="coolwarm", ax=ax
            )

        # Remove legend if seaborn creates one
        if ax.get_legend():
            ax.get_legend().remove()

        ax.set_title(var, fontsize=18)
        ax.set_xlabel("")
        ax.set_ylabel("")

        ax.tick_params(axis='both', labelsize=12)
    
    fig.supxlabel("Group", fontsize=16, y=0.08)


    plt.savefig(o_file)
    plt.show()


def correlation_matrix(df, o_file):
    # Matriu correlació
    corr_matrix = df.corr(numeric_only=True)

    plt.figure(figsize=(20, 15))
    ax = sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, center=0,  annot_kws={"size": 17})
    plt.title("Correlation matrix of dataset features", fontsize=20)
    plt.xticks(fontsize=18, rotation=90)
    plt.yticks(fontsize=18, rotation=0)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)
    plt.savefig(o_file)
    plt.show()

    return

def remove_outliers(df_group, numeric_variables, num_sd=4):
    df_group = df_group.reset_index()
    for var in numeric_variables:
        mean = df_group[var].mean()
        sd = df_group[var].std()
        outliers = (df_group[var] < mean - sd*num_sd) | (df_group[var] > mean + sd*num_sd)
        df_group.loc[outliers, var] = np.nan
    return df_group.dropna()

def knn_imputation(df):
    imputed_groups = {}

    for group_name, group_df in df.groupby("Group"):
        imputer = KNNImputer(n_neighbors=5)

        # Solo numéricas
        num = group_df.select_dtypes(include=[int, float])
        imputed_array = imputer.fit_transform(num)

        # Reconstruir el DF imputado
        imputed_df = group_df.copy()
        imputed_df[num.columns] = imputed_array

        imputed_groups[group_name] = imputed_df

    return pd.concat(imputed_groups.values()).sort_index()

def normalize_volumes(df, volumes, itv_var):
    df_norm = df.copy()
    for volume in volumes:
        df_norm[volume] = (df_norm[volume] / df_norm[itv_var]) * 100

    return df_norm

 # %%
## Load data
wdir = os.getcwd()
bbdd_dir = join(wdir, "BBDD")
project_name = "ENCLOSE"
results_dir = join(wdir,"results",project_name)
itv_var = "EstimatedTotalIntracranialVolume"

df = pd.read_excel(join(bbdd_dir,f"bbdd_{project_name}.xlsx"))
os.makedirs(results_dir, exist_ok=True)

# define volume variables
volume_vars = ['Lateral-Ventricle', 'Thalamus', 'Caudate', 'Putamen', 'Pallidum', 'Hippocampus', 'Amygdala',
               'VentralDC', 'CortexVol', 'Brain-Stem', 'WMH_volume']

##### EDA #####
df.info()

# check for 0
df.eq(0).sum()

# check for nan
df.isna().sum()

# Impute missing values
imputed_df = knn_imputation(df)
imputed_df.to_excel(join(bbdd_dir,f"imputed_bbdd_{project_name}.xlsx"),index=False)

### If you want data to be normalised run this:
# df_norm = normalize_volumes(imputed_df, volume_vars, itv_var)
# imputed_df = df_norm.copy()

# define numeric and demographic variables
numeric_variables = volume_vars + ['FA_global']
demographic_variables = ['Age', 'Sex']

# define variables to report
variables_to_report = demographic_variables + numeric_variables
variables_to_report.remove("Sex")

# sumamry of numeric variables
table_summary_variables = variables_summary(imputed_df, variables_to_report)
table_summary_variables.to_excel(join(wdir, "results",project_name,"summary_numerical_variables.xlsx"))

# plot demographic variables
plot_demographic_variables(imputed_df, demographic_variables, join(wdir,"results",project_name,"demographic_variables.jpg"))

# plot density plots of variables
density_plot_variables(imputed_df, numeric_variables, join(wdir,"results",project_name,"density_plot_vars.jpg"))

# plot boxplots of variables
boxplots_variables(imputed_df, numeric_variables, join(wdir,"results",project_name,"boxplot_vars.jpg"))

# correlation and covariance matrices
correlation_matrix(imputed_df[numeric_variables], join(wdir,"results",project_name,"corrmat_vars.jpg"))
