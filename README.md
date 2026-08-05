# Customer Segmentation using K-Means Clustering

A Machine Learning project that applies the **K-Means Clustering** algorithm to segment mall customers based on their **Annual Income** and **Spending Score**. The identified customer groups can help businesses design personalized marketing campaigns and improve customer engagement.

---

## Project Overview

Customer segmentation is an important business strategy that helps organizations understand different types of customers based on their purchasing behavior. Since the dataset does not contain predefined labels, this project uses **unsupervised machine learning** with the **K-Means Clustering** algorithm.

The workflow includes:

- Data Collection
- Data Exploration
- Data Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Selection
- Elbow Method
- K-Means Clustering
- Model Evaluation using Silhouette Score
- Cluster Visualization
- Business Insights

---

## Dataset

**Dataset Name:** Mall Customer Segmentation Dataset

**Source:**
https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

### Dataset Features

| Feature | Description |
|----------|-------------|
| CustomerID | Unique customer ID |
| Gender | Male/Female |
| Age | Customer age |
| Annual Income (k$) | Annual income in thousand dollars |
| Spending Score (1-100) | Customer spending score assigned by the mall |

Number of Records: **200**

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- Git
- GitHub
- VS Code

---

## Project Structure

```
customer-segmentation-kmeans/
│
├── data/
│   ├── Mall_Customers.csv
│   └── Mall_Customers_Clustered.csv
│
├── notebooks/
│   └── Customer_Segmentation.ipynb
│
├── images/
│   ├── gender_distribution.png
│   ├── age_distribution.png
│   ├── income_distribution.png
│   ├── spending_distribution.png
│   ├── pairplot.png
│   ├── correlation_heatmap.png
│   ├── elbow_method.png
│   └── customer_clusters.png
│
├── report/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Installation

Clone the repository

```bash
git clone git@github.com:tess-ahh/customer-segmentation-kmeans.git
```

Move into the project directory

```bash
cd customer-segmentation-kmeans
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate the virtual environment

Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook

```bash
jupyter notebook
```

Open

```
notebooks/Customer_Segmentation.ipynb
```

Run all cells.

---

## Data Preprocessing

The following preprocessing steps were performed:

- Checked dataset dimensions
- Verified data types
- Checked missing values
- Checked duplicate records
- Generated descriptive statistics
- Selected relevant features for clustering

---

## Exploratory Data Analysis

The following visualizations were created:

- Gender Distribution
- Age Distribution
- Annual Income Distribution
- Spending Score Distribution
- Pair Plot
- Correlation Heatmap

These visualizations provide insights into customer demographics and spending behavior before clustering.

---

## Feature Selection

The clustering algorithm uses only the following features:

- Annual Income (k$)
- Spending Score (1-100)

These two features provide a clear representation of customer purchasing patterns.

---

## Elbow Method

The Elbow Method was used to determine the optimal number of clusters.

The Within Cluster Sum of Squares (WCSS) was calculated for values of **K = 1 to 10**.

The elbow point was observed at:

**K = 5**

Therefore,

```
Number of Clusters = 5
```

---

## K-Means Clustering

The K-Means algorithm grouped customers into five clusters based on Annual Income and Spending Score.

Each customer was assigned a cluster label representing a unique customer segment.

---

## Model Evaluation

Evaluation Metric:

**Silhouette Score**

The Silhouette Score measures how well-separated the generated clusters are.

A higher value indicates better clustering quality.

---

## Results

The project successfully segmented customers into five meaningful groups.

The final visualization shows:

- Five customer clusters
- Cluster centroids
- Clear separation between customer groups

---

## Business Insights

The generated customer segments can help businesses:

- Identify premium customers
- Improve customer retention
- Design personalized marketing campaigns
- Offer targeted discounts
- Increase customer satisfaction
- Improve overall sales performance

---

## Future Improvements

Possible enhancements include:

- Using additional customer features
- Comparing with DBSCAN and Hierarchical Clustering
- Deploying the project using Streamlit
- Integrating real-time customer data
- Automated customer recommendation system

---

