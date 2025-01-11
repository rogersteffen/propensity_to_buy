## Overview

I'm demonstrating the various levels of sophistication (and perhaps accuracy) of a propensity to buy model. 

The data is sourced from Kaggle: https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/

## Business Problem

When using a high cost marketing channel it is far more profitable to target customers (or prospects) with a high likelihood of conversion.

Think of all the junk mail you immediately throw away!

A propensity model would be one of many steps, including:
* User graph/deduplication
* Address validation
* Assure a match-back strategy for in-store purchases
* A/B/Holdout testing

This repository solely focuses on the propensity model.

## Notebooks

In each notebook I intend to limit the scope of exploration to 1-2 topics or ideas.

Outline:
00 - Rules based - _not started_
01 - Using simple features with logistic regression & random forest
02 - Handling of outliers & standardizing features
03 - Tidy up some repeatable pieces
04 - Balance the data sets
05 - Introduce additional features
06 - Hyperparameter search
07 - Gradient boost
08 - More features (seasonality, price sensitivity, categories)
09 - Feed forward neural network

## My Motivation

Some feature engineering ideas:
* Personal seasonality - that is, some buyers will routinely purchase the same time each year ... events, season, birthdays, holidays
* Price sensitivity - an individuals price sensitivity relative to the current discounts offered
* (Very detailed) temporal aggregation (aka time based-feature engineering) - here increasing the number of features and using short time slices to see when model improvements taper off
* Training record augmentation
* Regression vs. Classification - allowing for more nuance such as time decayed sum of future transactions

Data preparation/pre-processing:
* Explore the impact of not balancing, under sampling, oversampling
* Explore the impact of no Winsorization & mild Winsorization versus no Winsorization (but just row wise outlier treatment)

Tools:
* At work we typically used Spark (so Spark dataframes, MLlib), I wanted to try out Polars & DuckDB
* PyTorch for feed forward neural network





