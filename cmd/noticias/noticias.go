package main

import (
	"fmt"
	"github.com/Giulianos/ml-naive-bayes-classifier/classifier"
	"strings"
)

func toExample(text string) classifier.Example {
	example := make(classifier.Example)
	for _, word := range strings.Split(text, " ") {
		example[strings.ToLower(word)] = "1"
	}
	return example
}

func toExamples(texts []string) []classifier.Example {
	var examples []classifier.Example

	for _, text := range texts {
		examples = append(examples, toExample(text))
	}

	return examples
}

func main() {

	var texts = []string{
		"The government shutdown",
		"Federal employees are protesting shutdown",
		"Turn melancholy forth to funerals",
	}

	var classifications = []string{
		"news",
		"news",
		"poetry",
	}

	// Create classifier passing prior class probability
	nb := classifier.NewNaiveBayes(map[string]float64{
		"news":   0.5,
		"poetry": 0.5,
	})

	// Examples
	examples := toExamples(texts)
	fmt.Println(examples)

	// Train the classifier
	nb.Train(examples, classifications)

	// Estimate text category
	category, prioriProb := nb.Predict(toExample("The shutdown affects federal employees benefit"))

	fmt.Printf("The text corresponds to %s (%f)\n", category, prioriProb)

}
