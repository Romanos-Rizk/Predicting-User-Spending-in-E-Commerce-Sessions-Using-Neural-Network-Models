# Kaggle Competition: Predicting User Spending on E-Commerce Website

## Project Overview

The goal of this competition was to develop a machine learning system capable of accurately predicting user spending based on session events collected from an e-commerce website. The dataset includes various features related to user sessions and interactions with the site. The challenge was to identify the optimal preprocessing techniques, features, and model parameters to achieve the best overall performance.

## Data Sources

The dataset contains session events with the following features:
- `user_id`: Unique user identifier.
- `session_id`: Unique session identifier.
- `session_start_time`: When the user lands on the website.
- `session_expiry_time`: When the session expires.
- `event_time`: Timestamp of the current event within a session.
- `event_time_zone`: Time zone of the current event.
- `event_type`: Type of event (e.g., loading a page, changing the cart).
- `page_type`: Page type the user is viewing (e.g., category page, product page, cart page).
- `offer_decline_count`: Number of times the user declined an offer.
- `user_status`: User status (New Customer, Prior Browser, Prior Buyer).
- `cart_quantity`: Number of items in the cart.
- `cart_total`: Total amount of the current cart.
- `cart_data`: List of dictionaries with product information.
- `last_offer_type`: Type of the last offer received.
- `last_reward_value`: Discount value of the last offer.
- `last_spend_value`: Amount required to collect the reward.
- `offer_display_count`: Number of offers received during the session.
- `user_screen_size`: Pixel resolution of the device.
- `offer_acceptance_state`: Whether the offer was accepted, declined, or ignored.
- `total`: Target variable indicating the total amount spent during the session.

## Tools & Purpose

- **Python**: General-purpose programming language for data analysis and modeling.
- **scikit-learn**: Library for machine learning algorithms and preprocessing.
- **TensorFlow**: Framework for developing neural network models.
- **matplotlib**: Library for creating visualizations.
- **seaborn**: Statistical data visualization library based on matplotlib.

## Data Cleaning and Preparation

- **Removal of `offer_decline_count`**: Excluded due to its negligible influence on the model.
- **Handling Missing Values**: Dropped rows with missing values, accounting for approximately 4.15% of the dataset.
- **Session Duration**: Computed and added duration of each session.
- **Feature Extraction**: Extracted time-related features from `event_time` and added `day_range` for temporal analysis.
- **Normalization**: Normalized numerical columns for uniform scaling.
- **One-Hot Encoding**: Converted categorical columns into a numerical format.

## Exploratory Data Analysis

- Conducted preliminary analysis to understand feature distributions and relationships.
- Created visualizations to explore patterns in the dataset.

## Data Analysis

- **Neural Network Model**:
  - **Input Layer**: 64 neurons with ReLU activation.
  - **Hidden Layer**: 35 neurons with ReLU activation.
  - **Output Layer**: Single neuron with linear activation.
  - **Optimizer**: Adam optimizer.
  - **Loss Function**: Mean Squared Error (MSE).
  - **Epochs**: 116.
  - **Batch Size**: 850.
  - **RMSE Calculation**: Achieved RMSE close to 0 on both training and validation datasets, indicating excellent model performance.

## Results and Findings

- The neural network model outperformed other models tested, with RMSE close to 0.
- Traditional models such as Random Forest, KNN, and Gradient Boosting showed signs of overfitting.
- The neural network demonstrated superior generalization capabilities and accuracy in predicting user spending.

## Recommendations

- Implement the neural network model for predicting user spending to enhance user targeting and improve e-commerce platform performance.
- Consider exploring real-time analytics and advanced feature engineering techniques to further refine the model.

## Limitations

- Variability in prediction results due to random initialization and the lack of a fixed random state during subsetting.
- Challenges with handling large datasets during model fitting and grid searches.

## References

- [Kaggle Competition Page](https://www.kaggle.com/c/your-competition)
- [Relevant Papers on E-Commerce Prediction](https://link-to-relevant-papers)
