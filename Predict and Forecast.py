import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import csv

# Newton's Divided-Difference Interpolation
def divided_difference(x, y, xi):
    n = len(x)
    coef = np.zeros((n, n))
    coef[:, 0] = y

    # Calculate divided differences
    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (x[i + j] - x[i])

    # Interpolate value at xi
    result = coef[0][0]
    for i in range(1, n):
        term = coef[0][i]
        for j in range(i):
            term *= (xi - x[j])
        result += term
    return result

# Linear Regression to predict missing and forecast future values
def linear_regression(x, y, k):
    model = LinearRegression()
    x_known = np.array([i for i in range(len(y)) if y[i] is not None]).reshape(-1, 1)
    y_known = np.array([yi for yi in y if yi is not None])
    model.fit(x_known, y_known)

    # Predict missing values
    y_pred = np.copy(y)
    for i in range(len(y)):
        if y[i] is None:
            y_pred[i] = model.predict([[i]])[0]

    # Forecast future values (extrapolation)
    future_x = np.array([len(y) + i for i in range(1, k + 1)]).reshape(-1, 1)
    future_y = model.predict(future_x)

    return y_pred, future_y

# Visualization
def visualize_data(x, y_original, y_interpolated, y_forecasted=None):
    plt.plot(x, y_original, 'o-', label='Original Data', color='blue')
    plt.plot(x, y_interpolated, 'o-', label='Interpolated Data', color='green')
    
    if y_forecasted is not None:
        forecast_x = np.array([i for i in range(len(y_original), len(y_original) + len(y_forecasted))])
        plt.plot(forecast_x, y_forecasted, 'o-', label='Forecasted Data', color='red')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title("Data Visualization")
    plt.show()

# Function to load dataset from CSV
def load_dataset_from_csv():
    # print("ex: C:\Users\Christian\Desktop\MDS_FINAL_PROJECT")
    file_path = input("Enter the file path of the CSV file: ")
    x = []
    y = []

    try:
        with open(file_path, mode='r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  

            for row in csv_reader:
                x.append(float(row[0]))
                y_value = row[1].strip().lower()
                y.append(float(y_value) if y_value != "none" else None)

        return np.array(x), y
    except Exception as e:
        print(f"Error reading the file: {e}")
        return None, None

def main():
    while True:
        print("\nChoose a method:")
        print("[1] - Newton's Divided-Difference Interpolation")
        print("[2] - Linear Regression")
        print("[0] - Exit")
        choice = int(input("Enter your choice: "))

        if choice == 0:
            print("Exiting the program. Goodbye!")
            break

        # Load dataset from CSV file
        x, y = load_dataset_from_csv()
        if x is None or y is None:
            print("Invalid dataset. Please try again.")
            continue

        if choice == 1:
            # Newton's Divided-Difference Interpolation
            y_interpolated = y.copy()
            for i in range(len(y)):
                if y[i] is None:
                    y_interpolated[i] = divided_difference(
                        x[np.array([j for j in range(len(y)) if y[j] is not None])],
                        np.array([yi for yi in y if yi is not None]),
                        x[i]
                    )
            visualize_data(x, y, y_interpolated)
        
        elif choice == 2:
            # Simple Linear Regression
            k = int(input("Enter number of future points to forecast: "))
            y_interpolated, y_forecasted = linear_regression(x, y, k)
            visualize_data(x, y, y_interpolated, y_forecasted)

if __name__ == "__main__":
    main()
