package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"os"
)

func main() {
	// Create file reader
	file, _ := os.Open(os.Args[1])
	r := bufio.NewReader(file)
	tsvReader := csv.NewReader(r)
	tsvReader.Comma = '\t'
	tsvReader.LazyQuotes = true

	// Classes
	classes := make(map[string]int)

	// Read headers
	_, err := tsvReader.Read()

	if err != nil {
		log.Fatal(err)
	}

	for {
		record, err := tsvReader.Read()
		if err == io.EOF {
			break
		}
		if len(record) < 4 {
			continue
		}
		classes[record[3]]++
	}

	var total int

	for class, quantity := range classes {
		fmt.Printf("%s: %d\n", class, quantity)
		total += quantity
	}

	fmt.Printf("\nTotal: %d\n", total)
}
