package classifier

type _PDhKey struct {
	feature      string
	featureValue string
	class        string
}

type Example map[string]string

// NaiveBayes represents a Naive Bayes Classifier
type NaiveBayes struct {
	PDh map[_PDhKey]float64
	Ph  map[string]float64
}

func (nb NaiveBayes) conditionalPriorProbability(feature string, featureValue string, class string) float64 {
	key := _PDhKey{feature, featureValue, class}
	return nb.PDh[key]
}

func (nb NaiveBayes) classPriorProbability(class string) float64 {
	return nb.Ph[class]
}

func (nb NaiveBayes) posterioriProbability(example Example, class string) float64 {
	// TODO: use logarithms to calculate probabilities
	probability := nb.classPriorProbability(class)
	for feature, featureValue := range example {
		probability *= nb.conditionalPriorProbability(feature, featureValue, class)
	}

	return probability
}

// Predict receives an array of features and returns the predicted encoded class
func (nb NaiveBayes) Predict(example Example) (string, float64) {
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

	// Obtain relative frequencies
	for key, value := range freqTable {
		freqTable[key] = value / countTable[key.class]
	}

	nb.PDh = freqTable
}

// NewNaiveBayes creates a new Naive Bayes Classifier
func NewNaiveBayes(priorClassProbabilities map[string]float64) NaiveBayes {
	nb := NaiveBayes{}
	nb.Ph = priorClassProbabilities

	return nb
}
