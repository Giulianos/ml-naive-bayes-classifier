package classifier

import (
	"fmt"
)

type Metrics struct {
	ConfusionMatrix map[string]map[string]uint64
	Outcomes        uint64
	TP              map[string]uint64
	FP              map[string]uint64
	TN              map[string]uint64
	FN              map[string]uint64
	Accuracy        map[string]float64
	Precision       map[string]float64
	Recall          map[string]float64
	F1Score         map[string]float64
	TPRate          map[string]float64
	FPRate          map[string]float64
	classes         []string
}

func createConfusionMatrix(classes []string) map[string]map[string]uint64 {
	matrix := make(map[string]map[string]uint64, len(classes))
	for _, class := range classes {
		matrix[class] = make(map[string]uint64, len(classes))
	}

	return matrix
}

func (m *Metrics) buildOutcomes() {
	m.TP = make(map[string]uint64, len(m.classes))
	m.TN = make(map[string]uint64, len(m.classes))
	m.FP = make(map[string]uint64, len(m.classes))
	m.FN = make(map[string]uint64, len(m.classes))

	for _, class := range m.classes {
		if class == "" {
			continue
		}
		m.TP[class] = m.ConfusionMatrix[class][class]
		for _, class2 := range m.classes {
			if class2 == "" {
				continue
			}
			m.FN[class] += m.ConfusionMatrix[class][class2]
			m.FP[class] += m.ConfusionMatrix[class2][class]
		}
		m.FN[class] -= m.TP[class]
		m.FP[class] -= m.TP[class]
		m.TN[class] = m.Outcomes - m.TP[class] - m.FP[class] - m.FN[class]
	}
}

func (m *Metrics) computeAccuracies() {
	m.Accuracy = make(map[string]float64, len(m.classes))
	for _, class := range m.classes {
		m.Accuracy[class] = float64(m.TP[class]+m.TN[class]) / float64(m.TP[class]+m.TN[class]+m.FP[class]+m.FN[class])
	}
}

func (m *Metrics) computePrecisions() {
	m.Precision = make(map[string]float64, len(m.classes))
	for _, class := range m.classes {
		m.Precision[class] = float64(m.TP[class]) / float64(m.TP[class]+m.FP[class])
	}
}

func (m *Metrics) computeRecalls() {
	m.Recall = make(map[string]float64, len(m.classes))
	for _, class := range m.classes {
		m.Recall[class] = float64(m.TP[class]) / float64(m.TP[class]+m.TN[class])
	}
}

func (m *Metrics) computeF1Score() {
	m.F1Score = make(map[string]float64, len(m.classes))
	for _, class := range m.classes {
		m.F1Score[class] = 2 * m.Recall[class] * m.Precision[class] / (m.Recall[class] + m.Precision[class])
	}
}

func (m *Metrics) computeTPRate() {
	m.TPRate = make(map[string]float64, len(m.classes))
	for _, class := range m.classes {
		m.TPRate[class] = float64(m.TP[class]) / float64(m.TP[class]+m.FN[class])
	}
}

func (m *Metrics) computeFPRate() {
	m.FPRate = make(map[string]float64, len(m.classes))
	for _, class := range m.classes {
		m.FPRate[class] = float64(m.FP[class]) / float64(m.FP[class]+m.TN[class])
	}
}

// EvalClassifier evaluates a classifier with the provided test set
// the classifier is assumed to be already trained
func EvalClassifier(classifier Classifier, testExamples []Example, testClassification []string) Metrics {
	metrics := Metrics{}
	metrics.classes = classifier.GetClasses()
	metrics.ConfusionMatrix = createConfusionMatrix(metrics.classes)

	for index, example := range testExamples {
		actual := testClassification[index]
		got, _ := classifier.Classify(example)

		// Add result to confusion matrix
		metrics.ConfusionMatrix[actual][got]++
		metrics.Outcomes++
	}

	// Calculate TP, FP, TN, FN from Confusion Matrix
	metrics.buildOutcomes()

	// Calculate Accuracies
	metrics.computeAccuracies()

	// Calculate Precisions
	metrics.computePrecisions()

	// Calculate Recalls
	metrics.computeRecalls()

	// Calculate F1-Scores
	metrics.computeF1Score()

	// Calculate TP Rate
	metrics.computeTPRate()

	// Calculate FP Rate
	metrics.computeFPRate()

	return metrics
}

func (metrics Metrics) confusionMatrixToString() string {
	var rep string

	if metrics.ConfusionMatrix == nil {
		return ""
	}

	rep += "\t"

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
func (m Metrics) String() string {
	cm := fmt.Sprintf("%s\n", m.confusionMatrixToString())
	metrics := ""
	for _, class := range m.classes {
		if class == "" {
			continue
		}
		metrics += class + "\n"
		metrics += fmt.Sprintf("TP: %d\nFP: %d\nTN: %d\nFN: %d\n", m.TP[class], m.FP[class], m.TN[class], m.FN[class])
		metrics += fmt.Sprintf("Accuracy: %f\n", m.Accuracy[class])
		metrics += fmt.Sprintf("Precision: %f\n", m.Precision[class])
		metrics += fmt.Sprintf("Recall: %f\n", m.Recall[class])
		metrics += fmt.Sprintf("F1-Score: %f\n", m.F1Score[class])
		metrics += fmt.Sprintf("TP Rate: %f\n", m.TPRate[class])
		metrics += fmt.Sprintf("FP Rate: %f\n", m.FPRate[class])
	}

	return cm + metrics
}
