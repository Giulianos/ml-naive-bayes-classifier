package main

import (
  "fmt"
  "github.com/Giulianos/ml-naive-bayes-classifier/classifier"
)

func main() {
  nb := classifier.NewNaiveBayes()

  class, precision := nb.Predict([]uint64 { 1, 0, 1, 0 })
  fmt.Printf("Pertenece a %d con %f de confidencia\n", class, precision)
}
