package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"github.com/Giulianos/ml-naive-bayes-classifier/classifier"
	"io"
	"log"
	"math"
	"os"
	"strings"
)

func loadDataSet(path string) ([]classifier.Example, []string, []string) {
	file, _ := os.Open(path)
	r := bufio.NewReader(file)
	tsvReader := csv.NewReader(r)
	tsvReader.Comma = '\t'
	tsvReader.LazyQuotes = true

	var examples []classifier.Example
	var classifications []string
	var line uint64 = 0

	// Read headers
	headers, err := tsvReader.Read()
	line++

	if err != nil {
		log.Fatal(err)
	}

	for {
		record, err := tsvReader.Read()
		line++
		if err == io.EOF {
			break
		}
		if len(record) != 4 {
			log.Printf("Error on %s (line %d)\n", path, line)
			continue
		}
		// Add classification
		classifications = append(classifications, record[len(record)-1])
		// Add example
		examples = append(examples, toExample(record[1]))
	}

	return examples, classifications, headers
}

func getCategoriesFrequencies(classifications []string) map[string]float64 {
	frequencies := map[string]float64{}

	// Count appearances of each category
	for _, class := range classifications {
		frequencies[class]++
	}

	// Obtain relative frequency
	for key, value := range frequencies {
		frequencies[key] = value / float64(len(classifications))
	}

	return frequencies
}

func toExample(text string) classifier.Example {
	example := make(classifier.Example)
	for _, word := range strings.Split(text, " ") {
		example[strings.ToLower(word)] = "1"
	}
	return example
}

func splitDataset(examples []classifier.Example, classifications []string, splitNum int) ([]classifier.Example, []string, []classifier.Example, []string) {
	testStart := splitNum
	testEnd := int(math.Min(float64(splitNum+len(examples)/10), float64(len(examples))))

	trainingExamples := append(examples[:testStart], examples[testEnd:]...)
	trainingClassifications := append(classifications[:testStart], classifications[testEnd:]...)

	testExamples := examples[testStart:testEnd]
	testClassifications := classifications[testStart:testEnd]

	return trainingExamples, trainingClassifications, testExamples, testClassifications
}

func evalClassifier(examples []classifier.Example, classifications []string, evalNum int) classifier.Metrics {
	// Split dataset
	trainingExamples, trainingClassif, testExamples, testClassif := splitDataset(examples, classifications, evalNum)

	// Get classes priori probability from training set
	prioriClassProb := getCategoriesFrequencies(trainingClassif)

	// Create classifier passing priori class probability
	nb := classifier.NewNaiveBayes(prioriClassProb)

	// Train the classifier
	nb.Train(trainingExamples, trainingClassif)

	// Eval classifier
	return classifier.EvalClassifier(nb, testExamples, testClassif)
}

func main() {
	// Load dataset
	examples, classifications, _ := loadDataSet(os.Args[1])

	var evaluations = make([]classifier.Metrics, 10)

	for i := range evaluations {
		evaluations[i] = evalClassifier(examples, classifications, i)

		fmt.Println(evaluations[i])
	}
}
