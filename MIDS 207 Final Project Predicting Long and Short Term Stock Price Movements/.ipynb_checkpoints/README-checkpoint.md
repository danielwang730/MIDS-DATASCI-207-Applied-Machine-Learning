# MIDS DATASCI 207 Final Project

The goal of this project is to **predict whether the Close price of a stock (at 3:59pm ET) is higher or lower than the Open price (at 9:30am ET) on the same day**.

While you can skip directly to the [final evaluation here](https://github.com/danielwang730/MIDS-DATASCI-207-Applied-Machine-Learning/blob/main/MIDS%20207%20Final%20Project%20Predicting%20Long%20and%20Short%20Term%20Stock%20Price%20Movements/06%20Final%20Model%20Evaluation/01_Final_Model_Evaluation.ipynb), it took substantial trial and error to reach this point. Below is a guide highlighting the most important notebooks and key sections to review. While this file summarizes the overall outcomes, **the notebooks themselves contain more detailed commentary and capture the reasoning behind my decisions at each step**. Lastly, you're welcome to explore the archived/original notebooks as well, though many of them contain exploratory or supplementary work that's not essential for understanding the main progression of my work.


### Step 0: Data Preprocessing & Feature Engineering

Our project begins in this [data preprocessing notebook](https://github.com/danielwang730/MIDS-DATASCI-207-Applied-Machine-Learning/blob/main/MIDS%20207%20Final%20Project%20Predicting%20Long%20and%20Short%20Term%20Stock%20Price%20Movements/00%20Data%20Processing%20and%20Feature%20Engineering/00_Data_Preprocessing.ipynb), where we downloaded and cleaned our dataset.

Next, we performed some [initial data visualizations](https://github.com/danielwang730/MIDS-DATASCI-207-Applied-Machine-Learning/blob/main/MIDS%20207%20Final%20Project%20Predicting%20Long%20and%20Short%20Term%20Stock%20Price%20Movements/00%20Data%20Processing%20and%20Feature%20Engineering/01_Data_Visualization.ipynb). However, visualizations based on raw stock prices proved to be less insightful due to the inherent variability across different tickers.

Lastly, we constructed derived features from the raw price and volume data in our [feature engineering notebook](https://github.com/danielwang730/MIDS-DATASCI-207-Applied-Machine-Learning/blob/main/MIDS%20207%20Final%20Project%20Predicting%20Long%20and%20Short%20Term%20Stock%20Price%20Movements/00%20Data%20Processing%20and%20Feature%20Engineering/02_Feature_Engineering.ipynb), and explained our reasoning for selecting specific technical indicators.

*Note: While we outlined these steps separately, we re-ran most of the feature engineering procedures in later notebooks to ensure consistency and correctness.*

---

### Step 1: [No Time Series Baseline Models](https://github.com/danielwang730/MIDS-DATASCI-207-Applied-Machine-Learning/blob/main/MIDS%20207%20Final%20Project%20Predicting%20Long%20and%20Short%20Term%20Stock%20Price%20Movements/01%20Baseline%20Models/00_No_Time_Series_Baseline_Models.ipynb)
*Jump to `In [43]:` in the notebook to skip the feature engineering steps described above.*

Our first baseline models ignore the time series nature of the data. We used logistic regression classification and linear regression by shuffling all indices and splitting the pooled data into training, validation, and test sets without regard for sequence or time order. We also performed some more EDA of our data here, and we analyzed the distribution of our features and outcomes.

Overall, this baseline model performed poorly, likely due to its assumption that each observation is independent of the others (which definitely does not hold in this context). A model with some form of memory or sequential awareness, capable of incorporating information from previous observations, would perform significantly better.

---

### Step 2: [Time Series Rolling Window Models (w/ Rolling OHLCV Values as Features)](https://github.com/danielwang730/MIDS-DATASCI-207-Applied-Machine-Learning/blob/main/MIDS%20207%20Final%20Project%20Predicting%20Long%20and%20Short%20Term%20Stock%20Price%20Movements/01%20Baseline%20Models/01_Time_Series_Baseline_Models.ipynb)
*Jump to `In [341]:` and continue to the end.*

In this next set of models, we aimed to incorporate the time-series nature of our data. To do this, we used the Open, High, Low, Close, and Volume (OHLCV) metrics from the previous 14 days as features for each prediction. This resulted in 70 features per observation (5 metrics × 14 days), where the first 5 correspond to the day prior to prediction, the next 5 to two days prior, and so on.

*We chose a 14-day window as it is a commonly used range in industry practice. This window can easily be adjusted as needed.*

**Importantly, we excluded metrics from the current day, as those values are not available until after the market closes. Including them would have given the model an unrealistic advantage by exposing information that wouldn’t be known during inference.**

To ensure the sequence of data remained intact, we verified that the dates were continuous. During this process, we discovered a ticker with missing days, which we ultimately removed to maintain sequence consistency.

We experimented with different configurations, including models trained on a single ticker, on multiple tickers, and with various dense layer setups. Across all experiments, the model struggled, and since the training and validation accuracies remained low, we saw no value in evaluating test accuracy further.

Still, this was a worthwhile exercise in testing a "pseudo-time-series" approach. It provided a clear contrast for future models, and it naturally led to our next candidate: an LSTM-based architecture.

---

### Step 3: [LSTM Models Using 80/20 Train/Test Split](https://github.com/danielwang730/MIDS-DATASCI-207-Applied-Machine-Learning/blob/main/MIDS%20207%20Final%20Project%20Predicting%20Long%20and%20Short%20Term%20Stock%20Price%20Movements/02%20Improved%20Models%20(with%20Normalization%20to%20Open)/00_LSTM_Models_80_20_Split.ipynb)
*Jump to `In [605]:` and continue to the end.*

LSTM (Long Short-Term Memory) models are well-suited for sequential financial data because they are designed to retain and leverage patterns over time. In this step, we implemented our first LSTM architecture using stock market data.

We began with a standard 80/20 train-test split, similar to our previous experiments. However, once again, we encountered poor test performance. The test accuracy was not only low but also comparable to the results from earlier models, suggesting that this modeling approach may have been fundamentally flawed for this dataset.

One hypothesis was the relatively large test size and lack of recent data, which we corrected for in the next section.

---

### Step 4: [LSTM Models Using 99/1 Train/Test Split](https://github.com/danielwang730/MIDS-DATASCI-207-Applied-Machine-Learning/blob/main/MIDS%20207%20Final%20Project%20Predicting%20Long%20and%20Short%20Term%20Stock%20Price%20Movements/02%20Improved%20Models%20(with%20Normalization%20to%20Open)/01_LSTM_Models_99_1_Split.ipynb)
*Jump to a bit before `In [527]:` and continue until the last section.*

In this second notebook, we adopted a **99/1 train-test split**. This approach provided two key benefits:  
1. It allowed the training data to be temporally closer to the test data, which would theoretically improve predictive performance.  
2. It increased the amount of data available for training, which would enhance the model’s learning capability overall.

In previous models, we split the dataset first, then created time series inputs. In this section, we reversed that process, building the time series **before** the split, to preserve time continuity.

Time series models like LSTMs require a window of prior data (e.g., 14 days) to make a prediction. If we split the data first, early test points won’t have the historical context needed for prediction (e.g., testing would start mid-December instead of December 1st). By constructing sequences before splitting, each test sample includes its necessary historical inputs.

A potential counterargument is that including pre-split data in the time series construction compromises the test set’s integrity by making it “seen.” However, we argue that this is **not data leakage** for the following reasons:
- We're not accessing the label (i.e., the target Close price) of the test data during training.
- Our goal is to simulate real-world forecasting, where you always have access to historical prices up to the current day.
- During deployment, input data from the previous days will naturally be available, and this process mirrors that reality.

Thus, we believe this strategy is both practical and valid for a time series forecasting task. It ensures that both training and testing examples are created using consistent methodology and that the test data reflects how the model would actually be used in practice.

---

### Step 4.5: Why the Series Data Needs Correction (Oversight #1)
*Continue from `In [602]:` in the same notebook as above.*

A critical limitation emerged from how the data was initially normalized. Specifically:

- All price data was normalized relative to the **Open price of each day**, meaning that each day’s Open was scaled to 1.0.
- Volume data was normalized using the **Previous Day’s Volume (PDV)**.

As a result, every day started with the same normalized price of 1.0, which effectively erased any notion of longer-term price movement. While large single-day jumps or gaps were still captured (via previous day features), the continuity and trend of price action across multiple days were lost. This ultimately undermined one of the primary strengths of LSTMs: learning from sequential dependencies over time.

#### Fixing Through Global Normalization by Ticker

To address this, we revised our preprocessing strategy in the next set of notebooks by applying global normalization (e.g., Min-Max scaling) across **all dates within each ticker**. This preserved relative price movements and trends while keeping tickers on a comparable scale. By maintaining the sequential nature of price changes, the LSTM would hopefully better learn from trends, momentum, and other long-term signals.

Despite the flawed normalization, however, the most recent LSTM models still achieved fairly good accuracy (excluding neutral cases). This may have suggested that 1) previous day features might have carried the most predictive value for next-day direction, and 2) technical indicators (SMA, RSI, NATR, etc.) could have been more informative than originally anticipated.

---

### Step 5: [LSTM Models With True Sequences Using 80/20 Train/Test Split](https://github.com/danielwang730/MIDS-DATASCI-207-Applied-Machine-Learning/blob/main/MIDS%20207%20Final%20Project%20Predicting%20Long%20and%20Short%20Term%20Stock%20Price%20Movements/03%20LSTM%20Models%20(with%20Corrected%20Sequence%20Normalization)/00_LSTM_Models_w_True_Sequences_80_20_Split.ipynb)
*Jump to a bit before `In [12]:` for an in-depth explanation of our updated feature scaling approach.*

In this notebook, we began using our **improved dataset** with proper **sequential normalization**. The LSTM models here are nearly identical to those from our earlier "Improved Models" notebooks in Step 3. The key difference is in how we normalized features: instead of normalizing price data to each day's Open and volume data to the Previous Day Volume (PDV), we applied global normalization per ticker across all available dates to their **respective maximums**, which ultimately preserved long-term temporal patterns.

Despite this improvement, the results were still underwhelming, as test accuracy remained significantly lower than training and validation accuracy. Similar to before, one likely reason was that the test data remained too far removed from the training data in time, and we didn't utilize as much training data as we could have. The next notebook addressed this by switching to a 99/1 split, which brought the test data closer to the training data temporally.

---

### Step 6: [LSTM Models With True Sequences Using 99/1 Train/Test Split](https://github.com/danielwang730/MIDS-DATASCI-207-Applied-Machine-Learning/blob/main/MIDS%20207%20Final%20Project%20Predicting%20Long%20and%20Short%20Term%20Stock%20Price%20Movements/03%20LSTM%20Models%20(with%20Corrected%20Sequence%20Normalization)/01_LSTM_Models_w_True_Sequences_99_1_Split.ipynb)
*Jump to `In [251]:` and continue to the end.*

This notebook applied the same LSTM architecture and corrected feature scaling, but with a **99/1 train/test split** to ensure the test data is more recent and better aligned with the training set. Once again, these models were nearly identical to the ones in Step 4.

Here, we finally observed promising results, as our most complex model — featuring multiple stocks, longer sequences, and minimal test/validation data — achieved a test accuracy of well above 50% and precision values above 75%, with much stronger performance in key classes (0 and 2). Simpler models also performed reasonably well, offering faster training times at a small cost to accuracy. These were helpful for rapid iteration and informed our next experiments with additional techniques like PCA.

**Note: While the absolute accuracy values turned out to be flawed due to a later-discovered oversight (explained in our final notebooks), the ***relative*** model performance remained meaningful and ultimately guided our final model selection.**

---

### Step 7: [PCA-LSTM Models](https://github.com/danielwang730/MIDS-DATASCI-207-Applied-Machine-Learning/blob/main/MIDS%20207%20Final%20Project%20Predicting%20Long%20and%20Short%20Term%20Stock%20Price%20Movements/04%20PCA%20LSTM%20Models/00_PCA_LSTM_Models.ipynb)
*Jump to `In [24]:` to see the PCA application and model performance.*

In this next set of experiments, we applied Principal Component Analysis (PCA) to our feature set with two primary goals: 1) reduce model training time, and 2) improve generalization by filtering out noise and minimizing overfitting.

We tested several LSTM configurations (e.g., hyperparameter tuning, simplified architectures), but we only kept the best-performing results. Surprisingly, PCA-based models underperformed compared to those trained on the original feature set. While PCA retained components explaining ~80% of the variance, it didn't translate to improved accuracy or noticeable speed gains. We believe the following factors explain this outcome:

1. Limited generalization benefit: Although PCA reduced our feature space to 13 components, our original LSTM models already included dropout layers, which provided sufficient generalization. As a result, PCA didn’t significantly reduce overfitting further.
2. Minimal speedup due to LSTM architecture: The bottleneck in LSTM models lies not in the input dimensionality, but in the **size and depth of the LSTM layers themselves**.

   - For example, a single LSTM layer with 256 units has (256 x 256 x 4 = 262,144 parameters) from hidden state connections alone.  
   - By contrast, the contribution from input features is much smaller (13 × 256 × 4 = 13,312 parameters).  

Thus, reducing input dimensionality with PCA has a minor effect on overall training time. Moreover, sequence length (e.g., 14 days) also contributes to computation time, since LSTMs process sequences sequentially, and this step cannot be simplified either or parallelized.

Overall, although PCA is a powerful tool in many contexts, its benefits were marginal in our LSTM setting. Given the accuracy trade-off and limited performance gains, we did not pursue PCA for our final model. That said, it remains worth exploring in future work or in combination with other model types better suited to dimensionality reduction.

---

### Step 8: [LSTM Models With Sliding Windows and Even Smaller Validation and Test Sizes](https://github.com/danielwang730/MIDS-DATASCI-207-Applied-Machine-Learning/blob/main/MIDS%20207%20Final%20Project%20Predicting%20Long%20and%20Short%20Term%20Stock%20Price%20Movements/03%20LSTM%20Models%20(with%20Corrected%20Sequence%20Normalization)/02_LSTM_Models_w_True_Sequences_Sliding_Windows.ipynb)
*Jump to `In [31]:` for the start of key experiments.*

This notebook built on our corrected sequential dataset by testing two ideas: (1) using extremely small validation and test sets to simulate real-world forecasting, and (2) evaluating a **sliding window** strategy for training and testing.

The thinking behind using an even smaller test set came from wanting to predict only the most recent day's close, and a sliding window framework allowed us to do that. Rather than training on all data at once, we simulated daily predictions by shifting a training window forward by one day at a time (e.g., train on days 1–98, validate on 99, test on 100; then slide forward to 2–99, 100, 101, etc.). We debated whether to retrain our models from scratch at each window or transfer learned weights across windows, but we chose the former due to time constraints and the latter's complexity.

We ultimately found that using limited training data (e.g., 100 days) significantly hurt performance, but expanding to at least a 700-day window (690 train, 3 validation, 7 test) improved results. In this case, we decided to include more than just one test day to see if we really needed to re-train the model every day, and surprisingly, test accuracies remained consistent across the 7 days. We extended the test set by 16 more days and saw no clear accuracy decay, suggesting the model might have picked up on durable patterns rather than date-specific trends.

**Note: Once again, due to the key issue explained below, the absolute accuracies reported here are flawed. Although we can still extract meaningful insights from the relative performance of different models, the reported results should not be taken at face value.**

---

### Step 8.5: [Why the Data Splits Need Correction (Oversight #2)](https://github.com/danielwang730/MIDS-DATASCI-207-Applied-Machine-Learning/blob/main/MIDS%20207%20Final%20Project%20Predicting%20Long%20and%20Short%20Term%20Stock%20Price%20Movements/05%20Final%20LSTM%20Models%20(with%20Corrected%20Train_Val_Test%20Data%20Splits)/00_Final_LSTM_Models_Correct_Data_Splits.ipynb)
*Start from the top, then jump to `In [320]:`.*

In all of our previous LSTM model notebooks, we unknowingly used a flawed version of the dataset. This likely contributed to the overly optimistic performance metrics we observed.

The core issue lies in the **ordering of our data** during the creation of our training, validation, and test splits. I mistakenly sorted the data by **Ticker first, then Date**, rather than the intended order of **Date first, then Ticker**. The likely reason as to why I missed this for so long was because after the creation of the data splits, they were converted from Pandas DataFrames to NumPy arrays, which made them much harder to check.

As a result, our test dataset ended up containing data **only from the last few tickers**, across all of their dates. For example, rather than using the most recent 10 days across all 475 stocks (which would have yielded 4,750 diverse test samples), we were effectively evaluating our models **only on the last few stocks alphabetically**, such as `ZTS`, across their entire date range.

This error fundamentally distorted what our test set was meant to represent: a **broad cross-section of recent market behavior across all tickers**.

#### Implication for LSTM Sequences and Previous Results

This ordering problem directly affected the sequences we created for training and testing our LSTM models. Since the model relied on **past** sequences to predict the current day’s outcome, feeding in **current** sequences (even from different tickers) undermined the generalizability of our evaluations. Therefore, our absolute results (i.e., accuracies, precisions, etc.) after Step 2 are likely **invalid**, since they evaluate performance on a skewed and non-representative subset of the data.

With that being said, **relative** performance and model efficiency should still stand, and our previous experiments still helped us identify our final model. Moreover, since our training and validation metrics were derived from a broad range of tickers, we expect some performance trends to still hold. For these reasons - and also due to time limitations and the desire to retain our original exploration process - we did not retroactively change the earlier notebooks to reflect this correction.

---

### Step 9: [Final LSTM Model Testing (with Corrected Training, Validation, and Test Data Splits)](https://github.com/danielwang730/MIDS-DATASCI-207-Applied-Machine-Learning/blob/main/MIDS%20207%20Final%20Project%20Predicting%20Long%20and%20Short%20Term%20Stock%20Price%20Movements/05%20Final%20LSTM%20Models%20(with%20Corrected%20Train_Val_Test%20Data%20Splits)/00_Final_LSTM_Models_Correct_Data_Splits.ipynb)
*Jump to `In [328]:` to see the results from the corrected implementation.*

After correcting the ordering issue in our dataset (ensuring that the data was split by **Date, then Ticker**, and not the reverse), we re-ran our LSTM models. The updated data split led to significantly worse results, which highlighted how much our previous models benefited from an incorrect (but easier) setup.

Also, despite using similar training data across all windows, test and validation performance varied dramatically by day, and some test days produced nearly unusable predictions. However, we did find that setting a prediction threshold (e.g., 0.50 for softmax confidence) improved performance, particularly for Class 2 (buy long). On these days, our model more consistently predicted the correct class, even if overall accuracy was still weak. Our predictions for Class 0 (sell short) were less reliable.

It’s worth noting that this original test window fell during a particularly volatile market period (12/23/24 to 12/31/24), which may have negatively impacted generalization. In the future, incorporating seasonality or macroeconomic indicators may improve performance during such periods.

Our final sliding window models (with corrected splits) offered mixed results. While precision for key classes was sometimes strong, the overall model still struggled, likely due to the loss of temporal proximity between training and test data. Furthermore, each run took ~30 minutes per test day, making this approach computationally expensive.

Nevertheless, we now have a more realistic benchmark. With corrected data, a clearer understanding of class-specific performance, and a tunable threshold, we’re ready to evaluate how well our model might perform on 2025 data.

**Note: These accuracy metrics only reflect directional correctness (i.e., predicting whether the close was higher, lower, or unchanged from the open). They do not capture the ***magnitude*** of that price movement, though we address this in the final notebook.**

---

### Step 10: [Final Model Evaluation](https://github.com/danielwang730/MIDS-DATASCI-207-Applied-Machine-Learning/blob/main/MIDS%20207%20Final%20Project%20Predicting%20Long%20and%20Short%20Term%20Stock%20Price%20Movements/06%20Final%20Model%20Evaluation/01_Final_Model_Evaluation.ipynb)
*I'd recommend skipping the bulk of the model training in the middle, but still going through the rest of the notebook. There is commentary throughout to guide analyses and decisions.*

This final notebook brings together everything we've worked on - our models, iterations, corrections, and insights - into a comprehensive evaluation.

**Important Note:** In all earlier LSTM model notebooks (except the previous one), we had been using an incorrectly processed dataset. This final evaluation was based only on the **corrected dataset**, where we fixed both our normalization method and the sequence ordering. Here, we evaluated our best-performing LSTM model using a sliding window approach, testing both on our original test set and on newly fetched 2025 data via `yfinance`. In total, we ended with 64 test days, with the final evaluation occurring on March 26, 2025.

#### Summary of Final Model and Evaluation Approach

- We implemented a flexible LSTM framework capable of training on any time window and evaluating on unseen data.
- Using sliding windows, we retrained the model for each new window and evaluated its respective test day.  
- Specifically, we extracted **precision metrics** for Class 0 (short) and Class 2 (long), and we then averaged these precision scores across all 64 test days to obtain our final results.
- To test real-world applicability, we simulated daily trades using only the data available at market open.
- On each day, we randomly selected **up to 10 stocks to trade**, based on the model’s long/short predictions.
- Across 1,000 random simulations, we consistently outperformed every baseline strategy.

#### Reflections & Results

When we began this project, we didn’t expect to build a model that could meaningfully outperform baseline strategies. The fact that it did - both statistically and in simulated financial performance - was surprising and encouraging.

While some of the apparent success could be attributed to current market conditions, we carefully validated our model's robustness by:
1. Evaluating it mathematically (via precision metrics and baseline comparisons),
2. Testing it in a realistic trading simulation, and
3. Comparing it against and beating every single baseline strategy we tried.

That said, we did make significant mistakes along the way:
1. **Improper normalization**: Our early models normalized features by the Open price of each day, rather than across all days for each ticker.
2. **Incorrect sequence ordering**: Data was initially ordered by Ticker → Date, instead of Date → Ticker, allowing test sequences to contain information from future days.

These errors cost us dozens of hours and multiple model re-trainings. However, we’ve now verified our entire data pipeline, re-downloaded and reprocessed the dataset from scratch, and re-tested the models with fixed logic. In the end, we are confident that this final notebook reflects the true capability of our approach.

#### Real-World Applicability

We believe our framework could reasonably be adapted to actual trading due to the following reasons:
1. It only uses features available at market open.
2. Models can be trained the night before and used the next day.
3. We limit our simulation to trading up to 10 stocks per day, making the setup more practical and grounded in real-world constraints.
4. The model's performance shows a steady, realistic edge - comparable to what one might expect in actual market conditions.

While more testing is needed - especially during volatile periods - our results suggest that long-term trading success is feasible.
