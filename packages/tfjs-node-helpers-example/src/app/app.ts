import {
  AccuracyMetricCalculator,
  BinaryClassificationTrainer,
  BinaryClassifier,
  BrierLossMetricCalculator,
  FNRMetricCalculator,
  FPRMetricCalculator,
  BinaryCrossentropyMetricCalculator,
  makeChunkedDataset,
  PRAUCMetricCalculator,
  PrecisionMetricCalculator,
  RecallMetricCalculator,
  ROCAUCMetricCalculator,
  SpecificityMetricCalculator,
  CohenKappaMetricCalculator,
  NPVMetricCalculator,
  MCCMetricCalculator,
  FBetaScoreMetricCalculator
} from '@ronas-it/tfjs-node-helpers';
import { data, layers, TensorContainer } from '@tensorflow/tfjs-node';
import { AgeFeatureEngineer } from './feature-engineers/age';
import { AnnualSalaryFeatureEngineer } from './feature-engineers/annual-salary';
import { GenderFeatureEngineer } from './feature-engineers/gender';
import { OwnsTheCarFeatureEngineer } from './feature-engineers/owns-the-car';
import { join } from 'node:path';
import { TrainingDataService } from './services/training-data';

export async function startApplication(): Promise<void> {
  await train();
  await predict();
}

async function train(): Promise<void> {
  const trainer = new BinaryClassificationTrainer({
    hiddenLayers: [layers.dense({ units: 128, activation: 'mish' }), layers.dense({ units: 128, activation: 'mish' })],
    inputFeatureEngineers: [
      new AgeFeatureEngineer(),
      new AnnualSalaryFeatureEngineer(),
      new GenderFeatureEngineer()
    ],
    outputFeatureEngineer: new OwnsTheCarFeatureEngineer(),
    metricCalculators: [
      new AccuracyMetricCalculator(),
      new PrecisionMetricCalculator(),
      new FBetaScoreMetricCalculator(1),
      new SpecificityMetricCalculator(),
      new RecallMetricCalculator(),
      new FNRMetricCalculator(),
      new FPRMetricCalculator(),
      new NPVMetricCalculator(),
      new MCCMetricCalculator(),
      new FBetaScoreMetricCalculator(2),
      new ROCAUCMetricCalculator(),
      new PRAUCMetricCalculator(),
      new BrierLossMetricCalculator(),
      new BinaryCrossentropyMetricCalculator(),
      new CohenKappaMetricCalculator()
    ]
  });

  const trainingDataService = new TrainingDataService({
    simulatedDelayMs: 100
  });

  await trainingDataService.initialize();

  const [validationSamplesCount, testingSamplesCount] = await Promise.all([
    trainingDataService.getValidationSamplesCount(),
    trainingDataService.getTestingSamplesCount()
  ]);

  const makeTrainingDataset = (): data.Dataset<TensorContainer> =>
    makeChunkedDataset({
      loadChunk: (skip, take) => trainingDataService.getTrainingSamples(skip, take),
      chunkSize: 32,
      batchSize: 32
    });

  const makeValidationDataset = (): data.Dataset<TensorContainer> =>
    makeChunkedDataset({
      loadChunk: (skip, take) => trainingDataService.getValidationSamples(skip, take),
      chunkSize: 32,
      batchSize: validationSamplesCount
    });

  const makeTestingDataset = (): data.Dataset<TensorContainer> =>
    makeChunkedDataset({
      loadChunk: (skip, take) => trainingDataService.getTestingSamples(skip, take),
      chunkSize: 32,
      batchSize: testingSamplesCount
    });

  const trainingDataset = makeTrainingDataset();
  const validationDataset = makeValidationDataset();
  const testingDataset = makeTestingDataset();

  await trainer.trainAndTest({
    trainingDataset,
    validationDataset,
    testingDataset,
    printTestingResults: true
  });

  await trainer.save(join(__dirname, './trained_model'));
}

async function predict(): Promise<void> {
  const classifier = new BinaryClassifier();

  await classifier.load(join(__dirname, './trained_model/model.json'));

  const ownsTheCar = await classifier.predict([0.2, 0.76, 0]);

  console.log('Predicted', ownsTheCar);
}
