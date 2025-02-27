# Weather Wiz: An AI-Powered Weather Forecasting System

## Abstract
Weather Wiz is an AI-driven framework developed to forecast ground temperature using 25 years of multi-station meteorological data provided by the Israel Meteorological Service (IMS). The system integrates a range of machine learning approaches—from traditional regularized regression models to advanced deep learning architectures such as Long Short-Term Memory networks (LSTM) and Graph Neural Networks (GNN). By capturing both temporal trends and spatial correlations, Weather Wiz delivers precise short-term forecasts. This paper details the problem formulation, data characteristics, preprocessing and exploratory analyses, modeling methodologies, and experimental results, while highlighting key insights obtained through extensive evaluations.

## 1. Introduction

Accurate weather forecasting is essential for numerous sectors, including agriculture, energy, transportation, and emergency planning. The Weather Wiz project focuses on predicting ground temperature (TG) by leveraging a rich dataset spanning 25 years of high-resolution weather measurements. Given the inherent complexity of meteorological phenomena—with influences from humidity, wind patterns, and precipitation—this work employs a combination of linear models, ensemble methods, and deep learning approaches. In particular, the integration of Graph Neural Networks allows the model to utilize spatial relationships among weather stations, thereby enhancing prediction accuracy.

## 2. Data Description

### 2.1 Data Source and Nature
The dataset is obtained from the [Israel Meteorological Service (IMS)](https://ims.gov.il), which records weather data from numerous stations distributed across Israel. The observations are captured every 10 minutes and later aggregated into hourly or daily summaries for analysis. This extensive temporal coverage and multi-station approach enable a comprehensive view of regional weather dynamics.

### 2.2 Data Extraction from the IMS API
Data extraction was performed by interfacing with the IMS API, which provides secure access to detailed meteorological data in JSON format. By specifying station identifiers and a defined date range, the system automatically queries and aggregates data on a monthly basis to overcome API limitations. The extracted data is then transformed into structured tabular formats for subsequent processing and analysis, ensuring comprehensive coverage of the 25-year period.

### 2.3 Features and Target Variable
- **Features:**
  - **Meteorological Variables:** Measurements such as relative humidity (RH), wind speed (WS), wind direction (WD), and rainfall (Rain).
  - **Engineered Features:** 
    - *Temporal Attributes:* Hour of day along with sine and cosine transformations to capture cyclical patterns.
    - *Wind Vectors:* Components derived from wind speed and wind direction that quantify directional wind influence.
- **Target:**  
  - **Ground Temperature (TG):** The primary variable to be predicted, representing the ground-level temperature.

### 2.4 Data Visualization
Several visualizations are used to explore the data:
- **Station Distribution:** A map showing the geographic locations of the IMS stations.
- **Feature Distributions:** Histograms displaying the distributions of key variables, such as TG and RH.
- **Time Series Trends:** Scatter plots and line graphs depicting the evolution of ground temperature over time.
- **Correlation Analysis:** A correlation heatmap reveals the relationships between various meteorological variables, highlighting potential predictors for temperature forecasting.

## 3. Data Preprocessing and Exploratory Data Analysis

### 3.1 Data Cleaning and Preparation
Data cleaning is a critical step in ensuring that the models are trained on high-quality, reliable data. Our cleaning process involved several key tasks:
- **Removal of Unrealistic Values:**  
  Data entries were filtered to enforce physical plausibility (e.g., ensuring \(0 \leq \text{RH} \leq 100\) and \(-15 \leq \text{TG} \leq 50\)). Extreme outliers and values that fell outside known meteorological ranges were discarded.
- **Handling Missing Values:**  
  Missing or null values in essential variables were identified and addressed. For some features, rows with missing data were dropped, while for others, imputation strategies were considered.
- **Datetime Conversion and Sorting:**  
  Timestamp strings were converted into datetime objects to facilitate chronological ordering. This ensured that subsequent time series analyses maintained temporal integrity.
- **Normalization and Scaling:**  
  Features were scaled using techniques such as MinMaxScaler to standardize their ranges, which is vital for many machine learning models to converge efficiently.
- **Feature Engineering:**  
  Additional features were computed to better capture underlying patterns in the data.

#### Mathematical Operations in Feature Engineering
To effectively represent cyclical and directional information, we applied the following mathematical transformations:
- **Cyclical Time Features:**  
  To capture the periodic nature of time (e.g., daily cycles), the hour of the day was transformed using sine and cosine functions:
  $$
  \text{hour}_{\sin} = \sin\left(\frac{2\pi \times \text{hour}}{24}\right)
  $$
  $$
  \text{hour}_{\cos} = \cos\left(\frac{2\pi \times \text{hour}}{24}\right)
  $$
  These transformations allow the model to recognize that, for instance, 23:00 and 01:00 are temporally close.
  
- **Wind Vector Components:**  
  To incorporate the directional component of wind, we computed the wind vector components from wind speed (\(WS\)) and wind direction (\(WD\)) as follows:
  $$
  wind_x = WS \times \cos\left(\frac{\pi \times WD}{180}\right)
  $$
  $$
  wind_y = WS \times \sin\left(\frac{\pi \times WD}{180}\right)
  $$
  These formulas convert the polar representation of wind into Cartesian coordinates, providing a richer representation of wind dynamics.

### 3.2 Exploratory Data Analysis (EDA)
EDA provides insights into the dataset’s structure and variability:
- **Distribution Analysis:**  
  Histograms and density plots illustrate the statistical distribution of temperature, humidity, and rainfall.
- **Time Series Visualization:**  
  Detailed time series plots highlight both short-term fluctuations and long-term trends in ground temperature.
- **Correlation Heatmaps:**  
  Visualizations of the correlation matrix help identify significant relationships among features, guiding feature selection and engineering.
- **Missing Value Patterns:**  
  Visualization of missing data patterns assists in understanding data quality and informs imputation strategies.

Additional EDA visualizations—such as a zoomed-in view of seasonal trends and scatter plots comparing multiple variables—offer deeper insights that drive the model development process.

## 4. Methodology

### 4.1 Modeling Approaches and Rationale
A multi-model strategy was adopted to address the diverse aspects of weather data. The reasons for selecting these specific models are as follows:
- **Lasso Regression:**  
  Chosen as a baseline model for its simplicity and interpretability. The L1 regularization inherent in Lasso promotes sparsity in the model coefficients, effectively identifying the most relevant features.
- **Random Forest:**  
  This ensemble method is capable of capturing complex nonlinear interactions among features. Its robustness and ability to handle high-dimensional data make it well-suited for weather prediction tasks.
- **Long Short-Term Memory (LSTM) Networks:**  
  LSTMs are designed to capture long-term dependencies in sequential data, making them ideal for modeling temporal dynamics in weather data. They effectively use historical sequences to predict future temperature values.
- **Graph Neural Networks (GNN):**  
  GNNs were chosen for their ability to integrate spatial information by modeling weather stations as nodes in a graph. By leveraging both spatial and temporal correlations, GNNs can capture localized weather patterns that traditional models may miss.

### 4.2 Explanation of Graph Neural Networks (GNN)
Graph Neural Networks (GNNs) represent a class of neural architectures that operate on graph-structured data. Unlike traditional neural networks that work on regular grids (such as images) or sequences (such as time series), GNNs are designed to handle irregular, interconnected data. Each node in the graph (in our case, a weather station) aggregates information from its neighboring nodes through a process known as message passing. During this process, each node updates its representation by combining its own features with those of its neighbors, often through learnable functions. This enables the network to capture both local interactions and the overall structure of the graph. In weather forecasting, this means that GNNs can effectively model spatial relationships between stations, providing a nuanced representation of regional weather patterns. The ability to incorporate additional node or edge features further enhances the model's performance, especially in environments where spatial dependencies are critical.

### 4.3 Training and Optimization

#### 4.3.1 Optimization Functions in the Models
Each model uses an optimization strategy tailored to its architecture:

- **LSTM Models – Adam Optimizer:**  
  The LSTM networks are trained using the Adam optimizer, which combines ideas from momentum and RMSProp to compute adaptive learning rates for each parameter. The update rules are given by:
  $$
  m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t,
  $$
  $$
  v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2,
  $$
  $$
  \hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t},
  $$
  $$
  w_{t+1} = w_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon},
  $$
  where \( g_t \) is the gradient at time \( t \), \( m_t \) and \( v_t \) are the first and second moment estimates, and \(\alpha\), \(\beta_1\), \(\beta_2\), and \(\epsilon\) are hyperparameters. This method allows for efficient convergence even in the presence of noisy gradients.

- **GNN Models – AdamW Optimizer:**  
  For training the GNN models, the AdamW optimizer is employed. AdamW decouples weight decay from the gradient-based update, which helps improve generalization. The weight update in AdamW is given by:
  $$
  w_{t+1} = w_t - \alpha \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda w_t \right),
  $$
  where \(\lambda\) is the weight decay coefficient. This decoupling ensures that the regularization term is applied directly to the weights, rather than being intertwined with the adaptive learning rate updates.

- **Lasso Regression and Random Forest:**  
  While Lasso Regression and Random Forest are not trained via gradient descent, they employ their own optimization methods. Lasso Regression uses coordinate descent to minimize the objective function:
  $$
  \min_{w} \; \|y - Xw\|_2^2 + \alpha \|w\|_1,
  $$
  which promotes sparsity in the coefficient vector \( w \). Random Forests, on the other hand, use recursive partitioning and impurity reduction (e.g., minimizing mean squared error) to construct decision trees, aggregating multiple trees to form the final ensemble prediction.

#### 4.3.2 Hyperparameter Tuning
Hyperparameters such as the number of hidden units, dropout rates, learning rates, and the number of epochs are tuned within specified ranges. For instance, the LSTM hidden units are varied between 32 and 128, while GNN hidden dimensions range from 32 to 256. Such tuning is critical for ensuring that the optimization algorithms perform efficiently and that the models converge to a robust solution.

### 4.4 Cross-Validation and Data Folds
To robustly evaluate model performance, a custom time series cross-validation strategy was employed. Unlike traditional random splits, this method preserves the temporal order of the data:
- **Chronological Splitting:**  
  The dataset is divided into sequential folds, where each fold consists of a training set (earlier time periods) and a test set (later time periods). This simulates a realistic forecasting scenario, where past data is used to predict future events.
- **Multiple Folds:**  
  By creating several folds, the model’s performance can be assessed across different time intervals. This helps in understanding the stability and consistency of the model’s predictions over time.
- **Avoiding Data Leakage:**  
  Sequential splitting ensures that no future information is inadvertently used during training, thereby preventing look-ahead bias and ensuring unbiased performance evaluation.

This cross-validation approach provides a realistic evaluation framework, allowing us to measure the model's predictive capabilities on unseen future data.

## 5. Experimental Results

### 5.1 Evaluation Metrics
The models are evaluated using several standard metrics:
- **Mean Absolute Error (MAE):** Measures the average absolute difference between the predicted and actual temperatures.
- **Mean Squared Error (MSE):** Provides a squared measure of prediction errors, penalizing larger deviations.
- **Coefficient of Determination (R² Score):** Indicates the proportion of variance in the target variable explained by the model.
- **Median Absolute Percentage Error (MdAPE):** Offers a relative measure of prediction accuracy.

### 5.2 Visualizations of Model Performance
A series of plots have been generated to illustrate the performance of the various models:
- **Time Series Plots:** Figures showing the actual versus predicted temperature values over different time segments.
- **Loss Curves:** Training and validation loss curves for the GNN model demonstrate convergence and learning stability.
- **Cross-Validation Summaries:** Bar charts comparing MAE, MSE, R², and MdAPE across multiple folds provide insights into model consistency.
- **Comparative Performance Charts:** Aggregated visual summaries highlight the strengths and weaknesses of each modeling approach.

*Example Figures:*
- **Figure 1:** Ground temperature prediction over time, showcasing actual and forecasted values.
- **Figure 2:** Loss curves from GNN training, illustrating both training and validation performance.
- **Figure 3:** Cross-validation performance summary across Lasso, Random Forest, LSTM, and GNN models.

### 5.3 Discussion of Results
The experimental results reveal that:
- **Lasso Regression** serves as a robust baseline with high interpretability.
- **Random Forest** captures complex nonlinearities but may suffer from higher variance in predictions.
- **LSTM Networks** effectively model temporal dependencies, particularly for short-term forecasts.
- **GNN Models** excel in incorporating spatial correlations, leading to improved accuracy in regions with dense station coverage.

The choice of model may vary based on the forecasting horizon and the specific application requirements. Overall, the integration of spatial information via GNNs demonstrates significant promise for enhancing predictive performance.

## 6. Discussion

The multi-model strategy employed in Weather Wiz underscores the importance of combining diverse methodological approaches when dealing with complex, high-dimensional weather data. While traditional linear and ensemble methods provide valuable baselines, deep learning models—especially those that incorporate temporal and spatial contexts—offer substantial improvements in forecasting accuracy. The experiments indicate that:
- Temporal dynamics are critical for short-term predictions, as captured by the LSTM.
- Spatial dependencies, often overlooked by conventional models, are effectively modeled by the GNN, particularly in geographically heterogeneous regions.

The comprehensive EDA and robust cross-validation strategies have ensured that the models generalize well on unseen data, although further refinements (e.g., additional feature engineering and more extensive hyperparameter tuning) could yield even better performance.

## 7. Conclusion

Weather Wiz demonstrates a robust and comprehensive approach to weather forecasting by integrating advanced machine learning techniques with detailed data preprocessing and exploratory analyses. By leveraging both temporal and spatial features, the system achieves a high level of accuracy in predicting ground temperature. Future work will focus on extending the framework to incorporate additional weather parameters, refining the spatial modeling aspects, and exploring longer-term forecasting horizons.

## 8. Future Work and Recommendations

- **Incorporation of Additional Metrics:** Extend the current system to predict other weather phenomena such as radiation, precipitation, and wind speed.
- **Enhanced Spatial Modeling:** Further refine the GNN architecture by incorporating dynamic edge weights based on real-time meteorological conditions.
- **Extended Forecast Horizons:** Experiment with longer sequence lengths and alternative temporal modeling techniques to support medium- and long-term forecasts.
- **Integration of External Data:** Explore the inclusion of satellite data and other remote sensing sources to enrich the feature set and improve model robustness.

## Acknowledgments
This research was made possible by the extensive data provided by the [Israel Meteorological Service (IMS)](https://ims.gov.il). We gratefully acknowledge the contributions of all researchers and practitioners whose work in machine learning and meteorology has paved the way for innovations like Weather Wiz.

---

*Note: This paper focuses on the scientific and methodological aspects of the project. For detailed technical implementation, code, and installation instructions, please refer to the supplementary documentation available in the project repository.*
