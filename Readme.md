# Stock Prediction

This repository contains a stock prediction program that utilizes LSTM (Long Short-Term Memory) models and the Alpha Vantage API to accurately forecast stock prices. By leveraging historical stock data and advanced techniques, this program offers reliable predictions and valuable insights for informed decision-making in the stock market.

## Features

- LSTM model: The program employs a powerful LSTM model, consisting of 4 layers, to capture complex patterns and trends in stock prices.
- Alpha Vantage API: The program integrates with the Alpha Vantage API to fetch real-time and historical stock data, ensuring accurate predictions.
- Training and testing data: The program intelligently splits the stock data, allocating 80% for model training and 20% for testing and evaluation.
- Loss function and optimizer: Mean squared error is used as the loss function, while the Adam optimizer optimizes the model for superior performance.
- Data scaling: The program applies MinMaxScaler to scale the data between 0 and 1, enhancing the accuracy of predictions.
- Dropout layer: To prevent overfitting, the program includes a Dropout layer in the model architecture, ensuring robust and reliable forecasts.
- Training visualization: The program utilizes the matplotlib library to plot the loss of the model during the training process, providing insights into model performance.
- Actual vs. predicted prices: The program generates visualizations that compare the actual stock prices with the predicted prices, allowing users to analyze trends and patterns.
- Future price prediction: Looking ahead, the program offers a glimpse into the future by plotting actual and predicted stock prices for the next 30 days, enabling users to anticipate market movements.

## Usage

1. Clone the repository:

```bash
git clone https://github.com/satyajit-2003/stock-prediction-program.git
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Obtain an API key from the [Alpha Vantage website](https://www.alphavantage.co/) and replace `YOUR_API_KEY` in the code with your actual API key.

4. Run the program:

```bash
python stock_prediction.py
```

5. Enter the stock symbol of your choice (e.g. AAPL, GOOGL, TSLA, etc.) and press Enter.

6. Explore the generated visualizations and gain insights into stock price predictions.

## Future Enhancements

- Implement a user-friendly GUI for an intuitive and interactive stock prediction experience.
- Include additional evaluation metrics to assess the performance of the model.
- Explore other advanced deep learning architectures for improved accuracy.

## Contributing

Contributions are welcome! Feel free to open issues and submit pull requests to enhance the functionality and usability of this stock prediction program. Please make sure to follow the established coding style and guidelines.

## License

This project is licensed under the [MIT License](https://opensource.org/license/mit/).

## Acknowledgements

- The Alpha Vantage API for providing access to financial market data.
- The matplotlib library for data visualization.
- The developers and contributors of the deep learning frameworks and libraries utilized in this project.

## Connect with Me

Connect with me on [LinkedIn](https://www.linkedin.com/in/satyajit-satapathy-45598b201/) to discuss this project and more! I would love to hear your feedback and suggestions.