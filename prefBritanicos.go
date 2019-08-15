package main

import (
  "fmt"
  "github.com/Giulianos/ml-naive-bayes-classifier/classifier"
)

func main() {
  nb := classifier.NewNaiveBayes([]float64 { 0.5, 0.5 })

  // Train the classifier
  nb.Train([][]uint64 {
    []uint64 { 0, 0, 1, 1, 1, 0 },
    []uint64 { 1, 1, 0, 0, 1, 0 },
    []uint64 { 1, 1, 0, 0, 0, 0 },
    []uint64 { 0, 1, 0, 0, 1, 0 },
    []uint64 { 0, 0, 0, 1, 0, 0 },
    []uint64 { 1, 0, 1, 1, 0, 0 },
    []uint64 { 1, 0, 0, 1, 1, 1 },
    []uint64 { 1, 1, 0, 0, 1, 1 },
    []uint64 { 1, 1, 1, 1, 0, 1 },
    []uint64 { 1, 1, 0, 1, 0, 1 },
    []uint64 { 1, 1, 0, 1, 1, 1 },
    []uint64 { 1, 0, 1, 1, 0, 1 },
    []uint64 { 1, 0, 1, 0, 0, 1 },
  })

  class, prioriProb := nb.Predict([]uint64 { 1, 0, 1, 1, 0 })

  fmt.Printf("Las preferencias corresponden a %d (%f)\n", class, prioriProb)

}
