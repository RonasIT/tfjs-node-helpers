import { Sample } from '../training/sample';
import { shuffle } from '../utils/shuffle';

export const splitSamplesIntoTrainingValidationTestForBinaryClassification = (
  samples: Array<Sample>,
  trainingPercentage: number = 70,
  validationPercentage: number = 15,
  testPercentage: number = 15
): {
  trainingSamples: Array<Sample>;
  validationSamples: Array<Sample>;
  testingSamples: Array<Sample>;
} => {
  if (trainingPercentage + validationPercentage + testPercentage !== 100) {
    throw new Error('trainingPercentage, validationPercentage and testPercentage don\'t add up to 100');
  }

  const shuffledSamples = shuffle(samples);
  const { truthySamples, falsySamples } = partitionSamplesIntoTruthyAndFalsy(shuffledSamples);

  const truthyTrainingSamplesCount = Math.floor(truthySamples.length * (trainingPercentage / 100));
  const falsyTrainingSamplesCount = Math.floor(falsySamples.length * (trainingPercentage / 100));
  const truthyValidationSamplesCount = Math.floor(truthySamples.length * ((trainingPercentage + validationPercentage) / 100));
  const falsyValidationSamplesCount = Math.floor(falsySamples.length * ((trainingPercentage + validationPercentage) / 100));

  const truthyTrainingSamples = truthySamples.slice(0, truthyTrainingSamplesCount);
  const falsyTrainingSamples = falsySamples.slice(0, falsyTrainingSamplesCount);
  const truthyValidationSamples = truthySamples.slice(truthyTrainingSamplesCount, truthyValidationSamplesCount);
  const falsyValidationSamples = falsySamples.slice(falsyTrainingSamplesCount, falsyValidationSamplesCount);
  const truthyTestingSamples = truthySamples.slice(truthyValidationSamplesCount);
  const falsyTestingSamples = falsySamples.slice(falsyValidationSamplesCount);

  const trainingSamples = shuffle(truthyTrainingSamples.concat(falsyTrainingSamples));
  const validationSamples = shuffle(truthyValidationSamples.concat(falsyValidationSamples));
  const testingSamples = shuffle(truthyTestingSamples.concat(falsyTestingSamples));

  return { trainingSamples, validationSamples, testingSamples };
};

const partitionSamplesIntoTruthyAndFalsy = (samples: Array<Sample>): {
  truthySamples: Array<Sample>;
  falsySamples: Array<Sample>;
} => {
  const truthySamples = [];
  const falsySamples = [];

  for (const sample of samples) {
    (sample.output[0] === 1)
      ? truthySamples.push(sample)
      : falsySamples.push(sample);
  }

  return { truthySamples, falsySamples };
};
