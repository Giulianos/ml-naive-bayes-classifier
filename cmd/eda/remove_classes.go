package main

import (
	"bufio"
	"encoding/csv"
	"io"
	"log"
	"os"
)

func main() {
	// Create file reader
	r := bufio.NewReader(os.Stdin)
	tsvReader := csv.NewReader(r)
	tsvReader.Comma = '\t'
	tsvReader.LazyQuotes = true

	// Create file writer
	w := bufio.NewWriter(os.Stdout)
	tsvWriter := csv.NewWriter(w)
	tsvWriter.Comma = '\t'

	// headers
	headerRecord, rerr := tsvReader.Read()
	werr := tsvWriter.Write(headerRecord)

	if rerr != nil {
		log.Fatal(rerr)
	}
	if werr != nil {
		log.Fatal(werr)
	}

	for {
		record, rerr := tsvReader.Read()
		if rerr == io.EOF {
			break
		}
		if len(record) < 4 {
			continue
		}

		for _, class := range os.Args[1:] {
			if record[3] == class {
				continue
			}
			werr = tsvWriter.Write(record)
		}
	}
}
