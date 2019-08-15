package classifier

type PDhKey struct {
	feature      uint64
	featureValue uint64
	class        uint64
}

// NaiveBayes
type NaiveBayes struct {
	PDh map[PDhKey]float64
	Ph  []float64
}

func (nb NaiveBayes) conditionalPriorProbability(feature uint64, featureValue uint64, class uint64) float64 {
	key := PDhKey{feature, featureValue, class}
	return nb.PDh[key]
}

func (nb NaiveBayes) classPriorProbability(class uint64) float64 {
	return nb.Ph[class]
}

func (nb NaiveBayes) posterioriProbability(features []uint64, class uint64) float64 {
	var probability float64 = nb.classPriorProbability(class)
	for feature, featureValue := range features {
		probability *= nb.conditionalPriorProbability(uint64(feature), featureValue, class)
	}

	return probability
}

// Predict receives an array of features and returns the predicted encoded class
func (nb NaiveBayes) Predict(features []uint64) (uint64, float64) {
	var maxArg uint64
	var maxVal float64
        var total float64
	for class, _ := range nb.Ph {
		val := nb.posterioriProbability(features, uint64(class))
                total += val
		if val > maxVal || class == 0 {
			maxArg = uint64(class)
			maxVal = val
		}
	}

	return maxArg, maxVal/total
}

func NewNaiveBayes() NaiveBayes {
  nb := NaiveBayes{}
  nb.PDh = map[PDhKey]float64{
    PDhKey{0, 1, 1}: 0.95,
    PDhKey{0, 0, 1}: 0.05,
    PDhKey{1, 1, 1}: 0.05,
    PDhKey{1, 0, 1}: 0.95,
    PDhKey{2, 1, 1}: 0.02,
    PDhKey{2, 0, 1}: 0.98,
    PDhKey{3, 1, 1}: 0.20,
    PDhKey{3, 0, 1}: 0.80,
    PDhKey{0, 1, 0}: 0.03,
    PDhKey{0, 0, 0}: 0.97,
    PDhKey{1, 1, 0}: 0.82,
    PDhKey{1, 0, 0}: 0.18,
    PDhKey{2, 1, 0}: 0.34,
    PDhKey{2, 0, 0}: 0.66,
    PDhKey{3, 1, 0}: 0.92,
    PDhKey{3, 0, 0}: 0.08,
  }
  nb.Ph = []float64 { 0.9, 0.1 }

  return nb
}
