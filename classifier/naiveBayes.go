package classifier

import (
	"math"
)

type _PDhKey struct {
	feature      string
	featureValue string
	class        string
}

// NaiveBayes represents a Naive Bayes Classifier
type NaiveBayes struct {
	PDh map[_PDhKey]float64
	Ph  map[string]float64
}

func (nb NaiveBayes) conditionalPriorProbability(feature string, featureValue string, class string) float64 {
	key := _PDhKey{feature, featureValue, class}
	prob, ok := nb.PDh[key]

	if ok {
		return prob
	} else {
		nonExistentFeatureKey := _PDhKey{"", "", class}
		return nb.PDh[nonExistentFeatureKey]
	}
}

func (nb NaiveBayes) classPriorProbability(class string) float64 {
	return nb.Ph[class]
}

func (nb NaiveBayes) posterioriProbability(example Example, class string) float64 {
	probability := math.Log(nb.classPriorProbability(class))
	for feature, featureValue := range example {
		probability += math.Log(nb.conditionalPriorProbability(feature, featureValue, class))
	}

	return math.Exp(probability)
}

// Predict receives an array of features and returns the predicted encoded class
func (nb NaiveBayes) Classify(example Example) (string, float64) {
	var maxArg string
	var maxVal float64
	var maxSet = false
	var total float64
	for class := range nb.Ph {
		val := nb.posterioriProbability(example, class)
		total += val
		if val > maxVal || !maxSet {
			maxArg = class
			maxVal = val
			maxSet = true
		}
	}

	return maxArg, maxVal / total
}

func laplaceCorrection(count float64, total float64, classes float64) float64 {
	return (count + 1) / (total + classes)
}

func (nb NaiveBayes) GetClasses() []string {
	classes := make([]string, len(nb.Ph))

	for class := range nb.Ph {
		classes = append(classes, class)
	}

	return classes
}

// Train receives the dataset and trains the classifier
func (nb *NaiveBayes) Train(examples []Example, classifications []string) {
	freqTable := make(map[_PDhKey]float64)
	countTable := make(map[string]float64)

	// Fill the frequency table
	for row, example := range examples {
		class := classifications[row]
		countTable[class]++
		for feature, featureValue := range example {
			freqTable[_PDhKey{feature, featureValue, class}]++
		}
	}

	var totalClasses = len(countTable)

	// Add frequency to non existent features
	for class, totalCount := range countTable {
		freqTable[_PDhKey{"", "", class}] = laplaceCorrection(0, totalCount, float64(totalClasses))
	}

	// Obtain relative frequencies
	for key, value := range freqTable {
		freqTable[key] = laplaceCorrection(value, countTable[key.class], float64(totalClasses))
	}

	nb.PDh = freqTable
}

// NewNaiveBayes creates a new Naive Bayes Classifier
func NewNaiveBayes(priorClassProbabilities map[string]float64) NaiveBayes {
	nb := NaiveBayes{}
	nb.Ph = priorClassProbabilities

	return nb
}
