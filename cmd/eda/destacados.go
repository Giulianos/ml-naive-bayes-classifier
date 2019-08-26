package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"math/rand"
	"os"
)

func main() {
	// Create file reader
	file, _ := os.Open(os.Args[1])
	r := bufio.NewReader(file)
	tsvReader := csv.NewReader(r)
	tsvReader.Comma = '\t'
	tsvReader.LazyQuotes = true

	// Read headers
	_, err := tsvReader.Read()

	if err != nil {
		log.Fatal(err)
	}

	var printed int

	for {
		record, err := tsvReader.Read()
		if err == io.EOF {
			break
		}
		if len(record) < 4 {
			continue
		}
		if record[3] == "Noticias destacadas" {
			if rand.Float64() < 0.001 {
				fmt.Println(record[1])
				printed++
			}
		}
	}

	fmt.Printf("Printed\t%d\n", printed)
}
