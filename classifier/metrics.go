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
	classes []string
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
	metrics.classes = classifier.GetClasses()
	metrics.ConfusionMatrix = createConfusionMatrix(metrics.classes)

	for index, example := range testExamples {
		actual := testClassification[index]
		got, _ := classifier.Classify(example)

		// Add result to confusion matrix
		metrics.ConfusionMatrix[actual][got]++
	}

	return metrics
}

func (metrics Metrics) confusionMatrixToString() string {
	var rep string

	for _, colClass := range metrics.classes {
		if colClass == "" {
			continue
		}
		rep += fmt.Sprintf("%s\t", colClass)
	}

	rep += "\n"

	for _, rowClass := range metrics.classes {
		if rowClass == "" {
			continue
		}
		rep += fmt.Sprintf("%s\t", rowClass)
		for _, colClass := range metrics.classes {
			if colClass == "" {
				continue
			}
			rep += fmt.Sprintf("%d\t", metrics.ConfusionMatrix[rowClass][colClass])
		}
		rep += "\n"
	}

	return rep
}

// String returns the string representation of the metrics
func String(m Metrics) string {
	return fmt.Sprintf("%s", m.confusionMatrixToString())
}