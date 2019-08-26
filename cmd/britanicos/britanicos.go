package main

import (
	"bufio"
	"encoding/csv"
	"flag"
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
	csvReader := csv.NewReader(r)

	var examples []classifier.Example
	var classifications []string

	// Read headers
	headers, err := csvReader.Read()

	if err != nil {
		log.Fatal(err)
	}

	for {
		record, err := csvReader.Read()
		if err == io.EOF {
			break
		}
		var example = classifier.Example{}
		for field, fieldValue := range record[:len(record)-1] { // Omit nationality
			example[headers[field]] = fieldValue
		}
		// Add classification
		classifications = append(classifications, record[len(record)-1])
		// Add example
		examples = append(examples, example)
	}

	return examples, classifications, headers
}

func parseExample(headers []string, exampleStr string) classifier.Example {
	var example = classifier.Example{}
	for field, fieldValue := range strings.Split(exampleStr, ",") {
		example[headers[field]] = fieldValue
	}

	return example
}

func main() {

	// Load flags
	fileName := flag.String("f", "", "dataset to use")
	prefs := flag.String("p", "1,0,1,1,0", "preferences to predict")

	flag.Parse()

	if *fileName == "" {
		flag.Usage()
		return
	}

	// Create classifier passing prior class probability
	nb := classifier.NewNaiveBayes(map[string]float64{
		"I": 0.5,
		"E": 0.5,
	})

	// Load dataset from file
	examples, classifications, headers := loadDataSet(*fileName)

	// Train the classifier
	nb.Train(examples, classifications)

	// Predict the nationality based on the passed preferences
	class, prioriProb := nb.Classify(parseExample(headers, *prefs))

	fmt.Printf("The preferences corresponds to %s (%f)\n", class, prioriProb)

}
