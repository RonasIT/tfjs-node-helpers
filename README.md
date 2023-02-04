# tfjs-node-helpers

## Introduction

This library was created to simplify the work with [TensorFlow.js][1] in
[Node][2].

Currently, this library provides the helpers for the
[binary classification task][3].  
The long-term plan is to implement helpers for other tasks, for example,
[regression][4] and [multiclass classification][5]), as well as to cover
different [machine learning approaches][6].

## Installation

Before you start using the helpers in your project, you need to install the
[@ronas-it/tfjs-node-helpers][7] package:

```bash
npm install @ronas-it/tfjs-node-helpers --save
```

## Usage

### Feature extraction

Before training any model, you need to extract the valuable information
from your dataset. This information is usually called *features*.
This library provides a few helpers to streamline the process of feature extraction.

First, you need to define the feature extractors. In the example below we
extract the `gender` feature from the dataset item. For that we create a
`GenderFeatureExtractor` class extending the `FeatureExtractor` base class
provided by the library. Please note that feature extractors also encode the
extracted information as a number in the range between `0` and `1`, so that it
can be consumed when training the model.

```typescript
type DatasetItem = {
  id: number;
  gender: string;
  age: number;
  annual_salary: number;
  owns_the_car: number;
};

class GenderFeatureExtractor extends FeatureExtractor<DatasetItem, FeatureType> {
  public featureType = FeatureType.GENDER;

  public extract(item: DatasetItem): Feature<FeatureType> {
    return new Feature({
      type: this.featureType,
      label: item.gender,
      value: (item.gender === 'Male') ? 1 : 0
    });
  }
}
```

That's it! Now we can use the defined feature extractor to extract valuable
information from our dataset.

### Metrics

After your model has been trained it's important to evaluate it.
One way to do this is by analyzing *metrics*.
The library helps measure model performance by passing a list of
metric calculators to the model trainer.

We have a list of built-in metric calculators for popular metrics:
- AccuracyMetricCalculator
- PrecisionMetricCalculator
- RecallMetricCalculator
- SpecificityMetricCalculator
- F1ScoreMetricCalculator
- FNRMetricCalculator
- FPRMetricCalculator

You can implement your own `MetricCalculator`. In the example below, we define
a metric calculator for `precision`. For that we create a `PrecisionMetricCalculator`
class extending the `MetricCalculator` base class provided by the library and
implementing `calculate` method.

```typescript
class PrecisionMetricCalculator extends MetricCalculator {
  public calculate(trueValues: Float32Array, predictedValues: Float32Array): Metric {
    const { tp, fp } = new ConfusionMatrix(trueValues, predictedValues);

    return new Metric({
      title: 'Precision',
      value: tp / (tp + fp)
    });
  }
}
```

### Binary classification

This library provides two classes to train and evaluate binary classification
models:

1. `BinaryClassificationTrainer` – used for training and testing.
1. `BinaryClassifier` – used for evaluation.

#### Creating the trainer

Before training the model, you need to create an instance of the
`BinaryClassificationTrainer` class first and provide a few parameters:

- `batchSize` – the number of training samples in each batch.
- `epochs` – the maximum number of iterations that we should train the model.
- `patience` – the number of iterations after which the trainer will stop if
  there is no improvement.
- `hiddenLayers` – a list of hidden layers. You can also provide the custom
  model by using the optional `model` parameter instead.
- `inputFeatureExtractors` – a list of feature extractors to extract information
  that should be fed into the model as inputs.
- `outputFeatureExtractor` – the feature extractor to extract information that
  we want to predict.
- `metricCalculators` – a list of metric calculators that will be used during test stage.

An example can be found below:

```typescript
const trainer = new BinaryClassificationTrainer({
  batchSize: BATCH_SIZE,
  epochs: EPOCHS,
  patience: PATIENCE,
  hiddenLayers: [
    layers.dense({ units: 128, activation: 'mish' }),
    layers.dense({ units: 128, activation: 'mish' })
  ],
  inputFeatureExtractors: [
    new AgeFeatureExtractor(),
    new AnnualSalaryFeatureExtractor(),
    new GenderFeatureExtractor()
  ],
  outputFeatureExtractor: new OwnsTheCarFeatureExtractor(),
  metricCalculators: [
    new AccuracyMetricCalculator(),
    new PrecisionMetricCalculator(),
    new SpecificityMetricCalculator(),
    new FPRMetricCalculator()
  ]
});
```

#### Training and testing

To train the model, you need to call the `trainAndTest` method of the
instantiated `BinaryClassificationTrainer`.

You can pass the `data` parameter, and in this case trainer will extract
features from the provided dataset first. If you want something more customized,
then you can create the datasets for training, validation and testing manually,
and pass them as the `trainingDataset`, `validationDataset` and `testingDataset`
parameters.

You can also print the testing results by setting the `printTestingResults` to
`true`.

An example can be found below:

```typescript
await trainer.trainAndTest({
  data,
  printTestingResults: true
});
```

##### Loading data asynchronously

When working with large dataset, you might find out that the whole dataset
can't fit in memory. In this situation you might want to load the data in
chunks. To do this, you can define the asynchronous generators for
`trainingDataset`, `validationDataset` and `testingDataset`.

This library provides the `makeChunkedDataset` helper to make it easier to
create chunked datasets where chunks are controlled with `skip` and `take`
parameters.

`makeChunkedDataset` helper accepts the following parameters:

- `loadChunk` – an asynchronous function accepting the numeric `skip` and `take`
  parameters and returning an array of samples.
- `chunkSize` – the number of samples loaded per chunk.
- `batchSize` – the number of samples in each batch.

```typescript
const loadTrainingSamplesChunk = async (skip: number, take: number): Promise<Array<Sample>> => {
  // Your samples chunk loading logic goes here. For example, you may want to
  //   load samples from database, or from a remote data source.
};

const makeTrainingDataset = (): data.Dataset<TensorContainer> => makeChunkedDataset({
  loadChunk: loadTrainingSamplesChunk,
  chunkSize: 32,
  batchSize: 32
});

// You should also define similar functions for validationDataset and
//   trainingDataset. We omit this for the sake of brevity.

const trainingDataset = makeTrainingDataset();
const validationDataset = makeValidationDataset();
const testingDataset = makeTestingDataset();

await trainer.trainAndTest({
  trainingDataset,
  validationDataset,
  testingDataset,
  printTestingResults: true
});
```

#### Saving the model

To save the trained model, you need to call the `save` method of the
instantiated `BinaryClassificationTrainer` and pass the path where the model
should be saved:

```typescript
await trainer.save(join(__dirname, './trained_model'));
```

#### Creating the classifier

Before evaluating the model, you need to create an instance of the
`BinaryClassifier` class:

```typescript
const classifier = new BinaryClassifier();
```

#### Loading the trained model

To load the trained model, you need to call the `load` method of the
instantiated `BinaryClassifier` class and pass the path where the model json
file is located:

```typescript
await classifier.load(join(__dirname, './trained_model/model.json'));
```

#### Evaluation

To evaluate the trained model, you need to load it first, and then call the
`predict` method of the instantiated `BinaryClassifier` class and pass an array
of encoded inputs which will be fed into the model:

```typescript
const ownsTheCar = await classifier.predict([0.2, 0.76, 0]);
```

## Roadmap

- [x] Binary classification ([#1](https://github.com/RonasIT/tfjs-node-helpers/pull/1))
- [x] Asynchronously loaded datasets ([#14](https://github.com/RonasIT/tfjs-node-helpers/issues/14))
- [x] Feature normalization ([#5](https://github.com/RonasIT/tfjs-node-helpers/issues/5))
- [x] Custom metrics ([#18](https://github.com/RonasIT/tfjs-node-helpers/issues/18))
- [ ] Add more metrics ([#17](https://github.com/RonasIT/tfjs-node-helpers/issues/17))
- [ ] Refactor features ([#25](https://github.com/RonasIT/tfjs-node-helpers/issues/25))
- [ ] Task-oriented architecture ([#26](https://github.com/RonasIT/tfjs-node-helpers/issues/26))
- [ ] Categorical features ([#19](https://github.com/RonasIT/tfjs-node-helpers/issues/19))
- [ ] Multiclass classification ([#3](https://github.com/RonasIT/tfjs-node-helpers/issues/3))
- [ ] Image classification ([#4](https://github.com/RonasIT/tfjs-node-helpers/issues/4))
- [ ] Regression ([#2](https://github.com/RonasIT/tfjs-node-helpers/issues/2))
- [ ] Object detection ([#27](https://github.com/RonasIT/tfjs-node-helpers/issues/27))
- [ ] Uncertainty ([#15](https://github.com/RonasIT/tfjs-node-helpers/issues/15))
- [ ] Handle class imbalance problem ([#10](https://github.com/RonasIT/tfjs-node-helpers/issues/10))
- [ ] Automated tests ([#6](https://github.com/RonasIT/tfjs-node-helpers/issues/6))
- [ ] Continuous Integration ([#11](https://github.com/RonasIT/tfjs-node-helpers/issues/11))
- [ ] Add an example of queued feature extraction and evaluation ([#12](https://github.com/RonasIT/tfjs-node-helpers/issues/12))
- [ ] Add an example of storing the extracted features ([#13](https://github.com/RonasIT/tfjs-node-helpers/issues/13))
- [ ] Add more examples ([#8](https://github.com/RonasIT/tfjs-node-helpers/issues/8))
- [ ] API reference ([#9](https://github.com/RonasIT/tfjs-node-helpers/issues/9))
- [ ] Dashboard to visualize metrics over time ([#7](https://github.com/RonasIT/tfjs-node-helpers/issues/7))

## Contributing

Thank you for considering contributing to `tfjs-node-helpers` library! The
contribution guide can be found in the [Contributing guide][8].

## License

`tfjs-node-helpers` is licensed under the [MIT license][9].

[1]:https://www.tensorflow.org/js
[2]:https://nodejs.org
[3]:https://en.wikipedia.org/wiki/Binary_classification
[4]:https://en.wikipedia.org/wiki/Regression_analysis
[5]:https://en.wikipedia.org/wiki/Multiclass_classification
[6]:https://en.wikipedia.org/wiki/Machine_learning#Approaches
[7]:https://www.npmjs.com/package/@ronas-it/tfjs-node-helpers
[8]:CONTRIBUTING.md
[9]:LICENSE
