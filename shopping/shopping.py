import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py shopping.csv")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    evidence = []
    labels = []
    month_map = {
        "Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3, 
        "May": 4, "June": 5, "Jul": 6, "Aug": 7, 
        "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11
    }
    try:
        with open(filename, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)

            for row in reader:
                administrative = int(row["Administrative"])
                informational = int(row["Informational"])
                productRelated = int(row["ProductRelated"])
                operatingSystems = int(row["OperatingSystems"])
                browser = int(row["Browser"])
                region = int(row["Region"])
                trafficType = int(row["TrafficType"])

                month_str = row["Month"]
                month = month_map.get(month_str, -1)
                if month == -1:
                    print(f"Warning: Invalid month value '{month_str}' found. Defaulting to 0 (Jan).")
                    month = 0

                visitorType = 1 if row["VisitorType"] == "Returning_Visitor" else 0
                weekend = 1 if row["Weekend"] == "TRUE" else 0

                administrative_Duration = float(row["Administrative_Duration"])
                informational_Duration = float(row["Informational_Duration"])
                productRelated_Duration = float(row["ProductRelated_Duration"])
                bounceRates = float(row["BounceRates"])
                exitRates = float(row["ExitRates"])
                pageValues = float(row["PageValues"])
                specialDay = float(row["SpecialDay"])

                revenue = 1 if row["Revenue"] == "TRUE" else 0
                
                evidence.append([administrative, administrative_Duration, informational, informational_Duration, 
                                productRelated, productRelated_Duration, bounceRates, exitRates, pageValues, 
                                specialDay, month, operatingSystems, browser, region, trafficType, 
                                visitorType, weekend])
                labels.append(revenue)

        return (evidence, labels)
    
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")        


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    true_positives = 0  # TP: Actual 1, Predicted 1
    true_negatives = 0  # TN: Actual 0, Predicted 0
    false_positives = 0 # FP: Actual 0, Predicted 1
    false_negatives = 0 # FN: Actual 1, Predicted 0
    
    for actual, predicted in zip(labels, predictions):
        if actual == 1 and predicted == 1: true_positives += 1
        elif actual == 0 and predicted == 0: true_negatives += 1
        elif actual == 0 and predicted == 1: false_positives += 1
        elif actual == 1 and predicted == 0: false_negatives += 1

    actual_positives = true_positives + false_negatives # TP + FN
    actual_negatives = true_negatives + false_positives # TN + FP

    if actual_positives == 0: sensitivity = 0.0
    else: sensitivity = true_positives / actual_positives

    if actual_negatives == 0: specificity = 0.0
    else: specificity = true_negatives / actual_negatives
    
    return (sensitivity, specificity)
        

if __name__ == "__main__":
    main()
