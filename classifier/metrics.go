package classifier

import (
	"fmt"
)

type Metrics struct {
	ConfusionMatrix map[string]map[string]uint64
	Accuracy  float64
	Precision float64
	TPRate    float64
	F1Score float64
}

func createConfusionMatrix(classes []string) map[string]map[string]uint64 {
	matrix := make(map[string]map[string]uint64, len(classes))
	for _, class := range classes {
		matrix[class] = make(map[string]uint64, len(classes))
	}

	return matrix
}

// EvalClassifier evaluates a classifier with the provided test set
// the classifier is assumed to be already trained
func EvalClassifier(classifier Classifier, testExamples  []Example, testClassification []string) Metrics {
	metrics := Metrics{}
	metrics.ConfusionMatrix = createConfusionMatrix(classifier.GetClasses())

	for index, example := range testExamples {
		actual := testClassification[index]
		got, _ := classifier.Classify(example)

		// Add result to confusion matrix
		metrics.ConfusionMatrix[actual][got]++
	}

	return metrics
}

// String returns the string representation of the metrics
func String(m Metrics) string {
	return fmt.Sprintf("%s", m.ConfusionMatrix)
}