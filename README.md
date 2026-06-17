# volunteer-engagement-analysis
Python script analyzing volunteer engagement patterns for nonprofit communities
# Volunteer Engagement Analysis

A Python script that takes messy nonprofit volunteer data, cleans it, and 
produces a plain-language report on retention, engagement, and tenure patterns.

## What it does

- Generates realistic sample data with the kinds of problems real nonprofit 
  data actually has (duplicates, typos, mixed date formats, blank fields)
- Cleans everything through a series of small, commented functions
- Analyzes retention, engagement distribution, tenure cohorts, and regional 
  breakdowns
- Outputs a cleaned CSV, a chart, and a written insights report

## How to run it

python3 volunteer_engagement_analysis.py

No setup needed. It generates its own sample data and runs the full pipeline.

## Why this exists

Built as a portfolio artifact for the Claude Corps fellowship application. 
The analysis uses an industrial-organizational psychology lens — retention, 
tenure cohorts, and engagement patterns are the same constructs used to study 
how organizations keep and lose people.

## Built with

Python, pandas, numpy, matplotlib
