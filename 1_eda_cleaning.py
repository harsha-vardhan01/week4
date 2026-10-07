"""
Capstone Project - Part 1: Data Cleaning & Exploratory Data Analysis (EDA)
Dataset: Financial Sample (Segment, Country, Product, Sales, Profit, etc.)
"""

import pandas as pd

pd.set_option("display.width", 120)

# -----------------------------------------------------------------
# Load data
# -----------------------------------------------------------------
df = pd.read_excel("sales_data.xlsx")
df.columns = df.columns.str.strip()  # " Sales" had a leading space

print("=" * 70)
print("Shape:", df.shape)
print("=" * 70)
print(df.head(), "\n")
print(df.dtypes, "\n")

# -----------------------------------------------------------------
# Clean: missing values
# -----------------------------------------------------------------
print("Missing values:\n", df.isnull().sum(), "\n")

# 'Discount Band' is blank when no discount was applied - fill with
# 'No Discount' rather than dropping (these are valid, meaningful records).
# NOTE: avoid the literal string "None", since pandas re-reads it as NaN
# when the CSV is loaded back in later steps.
df["Discount Band"] = df["Discount Band"].fillna("No Discount")

print("Duplicate rows:", df.duplicated().sum(), "\n")

# -----------------------------------------------------------------
# EDA: key summary stats
# -----------------------------------------------------------------
print("=" * 70)
print("Overall summary")
print("=" * 70)
print(df[["Units Sold", "Sales", "Profit", "COGS", "Discounts"]].describe(), "\n")

print("=" * 70)
print("Sales & Profit by Segment")
print("=" * 70)
by_segment = df.groupby("Segment")[["Sales", "Profit"]].sum().sort_values("Sales", ascending=False)
by_segment["Profit Margin %"] = (by_segment["Profit"] / by_segment["Sales"] * 100).round(1)
print(by_segment, "\n")

print("=" * 70)
print("Sales & Profit by Country")
print("=" * 70)
by_country = df.groupby("Country")[["Sales", "Profit"]].sum().sort_values("Sales", ascending=False)
print(by_country, "\n")

print("=" * 70)
print("Sales & Profit by Product")
print("=" * 70)
by_product = df.groupby("Product")[["Sales", "Profit", "Units Sold"]].sum().sort_values("Sales", ascending=False)
print(by_product, "\n")

print("=" * 70)
print("Monthly sales trend")
print("=" * 70)
monthly = df.groupby(["Year", "Month Number"])["Sales"].sum().reset_index()
print(monthly, "\n")

print("=" * 70)
print("Impact of Discount Band on Profit Margin")
print("=" * 70)
by_discount = df.groupby("Discount Band")[["Sales", "Profit"]].sum()
by_discount["Profit Margin %"] = (by_discount["Profit"] / by_discount["Sales"] * 100).round(1)
print(by_discount, "\n")

# -----------------------------------------------------------------
# Save the cleaned dataset for the next steps (regression + dashboard)
# -----------------------------------------------------------------
df.to_csv("cleaned_sales_data.csv", index=False)
print("Saved cleaned_sales_data.csv")
