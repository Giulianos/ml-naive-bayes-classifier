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

func (nb *NaiveBayes) Train(ds [][]uint64) {
  freqTable := make(map[PDhKey]float64)
  countTable := make(map[uint64]float64)

  // Fill the frequency table
  for _, row := range ds {
    class := row[len(row)-1]
    countTable[class] += 1
    for feature, featureValue := range row[:len(row)-1] {
      freqTable[PDhKey{uint64(feature), featureValue, class}] += 1
    }
  }

  // Obtain relative frequencies
  for key, value := range freqTable {
    freqTable[key] = value / countTable[key.class]
  }

  nb.PDh = freqTable
}

func NewNaiveBayes(probH []float64) NaiveBayes {
  nb := NaiveBayes{}
  nb.Ph = probH

  return nb
}
