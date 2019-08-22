package classifier

type Example map[string]string

type Classifier interface {
	Classify(example Example) (string, float64)
	GetClasses() []string
}