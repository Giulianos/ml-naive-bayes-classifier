package main

import (
	"bufio"
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/Giulianos/ml-naive-bayes-classifier/classifier"
)

var natMapping = []string{"I", "E"}

func loadDataSet(path string) [][]uint64 {
	file, _ := os.Open(path)
	r := bufio.NewReader(file)
	csvReader := csv.NewReader(r)

	var ds [][]uint64

	// Discard header
	csvReader.Read()
	for {
		record, err := csvReader.Read()
		if err == io.EOF {
			break
		}
		var row []uint64
		for _, fieldValue := range record[:len(record)-1] { // Omit nationality
			temp, _ := strconv.ParseUint(fieldValue, 10, 64)
			row = append(row, temp)
		}
		// Convert nationality to 1/0
		var convNat uint64
		if record[len(record)-1] == natMapping[1] {
			convNat = 1
		}
		row = append(row, convNat)
		ds = append(ds, row)
	}

	return ds
}

func parsePrefs(prefs string) []uint64 {
	var ret []uint64
	for _, val := range strings.Split(prefs, ",") {
		temp, _ := strconv.ParseUint(val, 10, 64)
		ret = append(ret, temp)
	}

	return ret
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
	nb := classifier.NewNaiveBayes([]float64{0.5, 0.5})

	// Load dataset from file
	ds := loadDataSet(*fileName)

	// Train the classifier
	nb.Train(ds)

	// Predict the nationality based on the passed preferences
	class, prioriProb := nb.Predict(parsePrefs(*prefs))

	fmt.Printf("The preferences corresponds to %s (%f)\n", natMapping[class], prioriProb)

}
