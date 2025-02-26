# Machine_Learning & Deep Learning                  
                                                       
                                                       
                                                       
                                                       
(<sub>work in progress</sub>)

     Machine Learning (ML) algorithms are broadly categorized into 

     (1) Supervised

     (2) Unsupervised

     (3) Semi-supervised

     (4) Reinforcement Learning


 ## 🔹 1. Supervised Learning Algorithms
Supervised learning involves labeled data, where the model learns to map inputs to outputs.


- 📌 A. ***<ins> Regression Algorithms (Predict Continuous Values) </ins>*** : These algorithms predict numerical values (e.g., stock prices, house prices).

| ALGO | Description| 
| ---- | ---- | 
| `Linear & Multiple Linear Regression` | Finds a linear relationship between independent and dependent variables. | 
| `Polynomial Regression` | Extends linear regression with polynomial features. |
| `Ridge Regression` |  Uses L2 regularization to prevent overfitting. |
| `Lasso Regression` |  Uses L1 regularization to shrink some coefficients to zero. |
| `Elastic Net Regression` |  A combination of Ridge and Lasso regression. |
| `Support Vector Regression (SVR)` |  Uses support vectors to predict continuous values. | 
| `Decision Tree Regression` |  Uses tree-based splits for prediction. |
| `Random Forest Regression` | An ensemble of multiple decision trees. |
| `XGBoost Regression` |  A gradient boosting method for high-performance regression. |
| `LightGBM Regression` |  A faster gradient boosting alternative to XGBoost. |
| `CatBoost Regression` |  A boosting model optimized for categorical data. |
| `Bagging Regressor` |  Uses bootstrap aggregation of multiple regressors. |
| `AdaBoost Regressor` |  Adaptive boosting for regression tasks. |
| `Gradient Boosting Regressor (GBR)` |  Sequentially improves weak models. |
| `Stacking Regressor` |  Combines multiple models using a meta-model. |
| `Bayesian Regression` |  Uses probabilistic models to estimate predictions. |

___________________________________________________________________________________
- 📌 B. ***<ins>Classification Algorithms (Predict Discrete Categories)</ins>*** : These algorithms predict categorical labels (e.g., spam detection, image recognition).
  
| ALGO | Description |
| ---- | ---- |
| `Logistic Regression` | A linear model for binary classification. |
|`K-Nearest Neighbors (KNN)` | Classifies based on the majority vote of nearest neighbors. |
|`Support Vector Machines (SVM)` | Finds the best hyperplane to separate classes. |
|`Decision Trees` | Classifies data using tree-like splits. |
|`Random Forest Classifier` | Uses multiple decision trees for better accuracy. |
|`Gradient Boosting Classifier` | An ensemble method improving weak models iteratively. |
|`XGBoost Classifier` | An optimized gradient boosting method. |
|`LightGBM Classifier` | A fast and efficient boosting model. |
|`CatBoost Classifier` | A gradient boosting model optimized for categorical data. |
|`Bagging Classifier` | Uses bootstrap aggregation to reduce variance.|
|`AdaBoost Classifier` | Uses weighted models to improve weak learners.|
|`Stacking Classifier` | Combines multiple models using a meta-classifier.|
|`Voting Classifier` | Averages multiple model predictions (hard/soft voting).|
|`Naïve Bayes` | Uses Bayes’ theorem for probabilistic classification (e.g., Gaussian, Multinomial). |
|`Artificial Neural Networks (ANNs)` | Multi-layer networks for deep learning tasks. |


## 🔹 2. Unsupervised Learning Algorithms
Unsupervised learning deals with unlabeled data, finding hidden patterns.

- 📌 A. ***<ins>Clustering Algorithms (Group Similar Data Points) </ins>*** : Used in customer segmentation, anomaly detection, etc.
  
| ALGO | Description |
| ---- | ---- |
|`K-Means Clustering` | Groups data into k clusters based on similarity.|
|`Hierarchical Clustering` | Builds a tree of clusters using a bottom-up or top-down approach.|
|`DBSCAN (Density-Based Spatial Clustering)` | Detects clusters based on density.|
|`Mean Shift Clustering`| Groups data by finding density peaks.|
|`Gaussian Mixture Model (GMM)` | A probabilistic clustering model based on Gaussian distributions.|
|`Agglomerative Clustering` | A bottom-up hierarchical clustering method.|

____________________________________________________________________
- 📌 B. ***<ins>Dimensionality Reduction Algorithms (Reduce Features While Retaining Information)</ins>*** : Used for data visualization and preprocessing.
  
| ALGO | Description |
| ----  | ---- |
|`Principal Component Analysis (PCA)` | Projects data into a lower-dimensional space.|
|`t-Distributed Stochastic Neighbor Embedding (t-SNE)` | Preserves local structures for visualization.|
|`Linear Discriminant Analysis (LDA)` | Maximizes separability between classes.|
|`Autoencoders` | Neural networks that learn compressed representations of data.|
|`Factor Analysis` | Identifies hidden variables influencing observed data.|
|`Singular Value Decomposition (SVD)` | Matrix factorization technique for dimensionality reduction.|

______________________________________________________________________
- 📌 C. ***<ins> Anomaly Detection Algorithms (Find Outliers in Data)</ins>*** : Used in fraud detection, intrusion detection, etc.

| ALGO | Description |
| ---- | ---- |
|`Isolation Forest` | Detects anomalies by isolating data points.|
|`Local Outlier Factor (LOF)` | Measures the local density of data points.|
|`One-Class SVM` | A variation of SVM for anomaly detection.|
|`Autoencoders (for anomaly detection)` | Detect anomalies by reconstructing normal patterns.|
|`Elliptic Envelope` | Assumes data follows a Gaussian distribution.|


## 🔹 3. Semi-Supervised Learning Algorithms
Semi-supervised learning is used when only a small portion of data is labeled.

| ALGO | Description |
| ---- | ---- |
|`Self-training` | Uses a supervised model iteratively on labeled and unlabeled data.|
|`Co-training` | Uses multiple classifiers to improve labeling accuracy.|
|`Graph-Based Semi-Supervised Learning` | Uses graph structures to propagate labels.|
|`Pseudo-Labeling` | Assigns labels to unlabeled data based on a trained model.|


## 🔹 4. Reinforcement Learning Algorithms
Reinforcement learning (RL) is used for sequential decision-making.

| ALGO | Description |
| ---- | ---- |
|`Q-Learning` | A value-based RL algorithm using Q-values for actions.|
|`Deep Q-Network (DQN`) | Uses neural networks to approximate Q-values.|
|`SARSA (State-Action-Reward-State-Action)` | Similar to Q-learning but learns on-policy.|
|`Policy Gradient Methods` | Directly optimize policy functions for better actions.|
|`Actor-Critic Methods` | Combines value-based and policy-based methods.|
|`Proximal Policy Optimization (PPO)` | A stable and efficient RL algorithm.|
|`Deep Deterministic Policy Gradient (DDPG)` | Handles continuous action spaces.|
|`Monte Carlo Methods` | Estimates future rewards based on sampled episodes.|


## 🔹 5. Deep Learning Algorithms
Deep Learning models use neural networks for complex data like images, text, and speech.

- 📌 A. ***<ins>Convolutional Neural Networks (CNNs)</ins>*** : Used in image processing.

| ALGO | Description |
| ---- | ---- |
|`LeNet-5` | One of the first CNN architectures.|
|`AlexNet` |Improved deep CNN for image classification.|
|`VGG16/VGG19` | Deep networks with 16/19 layers.|
|`ResNet` | Uses skip connections to improve training.|
|`EfficientNet` | Optimized for high accuracy with fewer parameters.|

__________________________________________________________________________

- 📌 B. ***<ins>Recurrent Neural Networks (RNNs)</ins>*** : Used in sequential data like text and time series.
  
| ALGO | Description |
| ---- | ---- |
|`Vanilla RNN` | Basic recurrent network for sequences.|
|`Long Short-Term Memory (LSTM)` | Improved RNN to handle long-term dependencies.|
|`Gated Recurrent Unit (GRU)` | Similar to LSTM but computationally efficient.|

___________________________________________________

- 📌 C. ***<ins> Transformers (NLP Models) </ins>*** : Used in language modeling and text understanding.

| ALGO | Description |
| ---- | ---- |
|`BERT (Bidirectional Encoder Representations from Transformers)` | Contextual NLP model.|
|`GPT (Generative Pre-trained Transformer)` | Used in text generation.|
|`T5 (Text-to-Text Transfer Transformer)` | Converts tasks into a text format.|
|`Transformer-XL` | Handles long text sequences efficiently.|


⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐

