package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"github.com/Giulianos/ml-naive-bayes-classifier/classifier"
	"io"
	"log"
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

func main() {

	// Load training set
	trainExamples, trainClassif, _ := loadDataSet(os.Args[1])

	// Get classes priori probability from training set
	prioriClassProb := getCategoriesFrequencies(trainClassif)

	// Create classifier passing priori class probability
	nb := classifier.NewNaiveBayes(prioriClassProb)

	// Train the classifier
	nb.Train(trainExamples, trainClassif)

	if len(os.Args) > 2 {
		// Load test set
		testExamples, testClassif, _ := loadDataSet(os.Args[2])

		// Eval classifier
		metrics := classifier.EvalClassifier(nb, testExamples, testClassif)

		// Print results
		fmt.Print(classifier.String(metrics))
	}
}
