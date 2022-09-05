import { BinaryClassificationTrainer, BinaryClassifier } from '@ronas-it/tfjs-node-helpers';
import { layers } from '@tensorflow/tfjs-node';
import { AgeFeatureExtractor } from './feature-extractors/age';
import { AnnualSalaryFeatureExtractor } from './feature-extractors/annual-salary';
import { GenderFeatureExtractor } from './feature-extractors/gender';
import { OwnsTheCarFeatureExtractor } from './feature-extractors/owns-the-car';
import { join } from 'node:path';
import data from '../assets/data.json';

const EPOCHS = 1000;
const PATIENCE = 20;
const BATCH_SIZE = 32;

export async function startApplication() {
  await train();
  await predict();
}

async function train(): Promise<void> {
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
    outputFeatureExtractor: new OwnsTheCarFeatureExtractor()
  });

  await trainer.trainAndTest({
    data,
    printResults: true
  });

  await trainer.save(join(__dirname, './trained_model'));
}

async function predict(): Promise<void> {
  const classifier = new BinaryClassifier();

  await classifier.load(join(__dirname, './trained_model/model.json'));

  const ownsTheCar = await classifier.predict([0.2, 0.76, 0]);

  console.log('Predicted', ownsTheCar);
}
