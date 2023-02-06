import { data, TensorContainer } from '@tensorflow/tfjs-node';
import { splitSamplesIntoTrainingValidationTestForBinaryClassification } from '../data-splitting/training-validation-test-for-binary-classification';
import { makeDataset } from '../utils/make-dataset';
import { extractFeatures } from './extract-features';
import { FeatureEngineer } from './feature-engineer';

export const prepareDatasetsForBinaryClassification = async <D, T>({
  data,
  inputFeatureEngineers,
  outputFeatureEngineer,
  batchSize,
  trainingPercentage,
  validationPercentage,
  testingPercentage
}: {
  data: Array<D>;
  inputFeatureEngineers: Array<FeatureEngineer<T, D>>;
  outputFeatureEngineer: FeatureEngineer<T, D>;
  batchSize: number;
  trainingPercentage?: number;
  validationPercentage?: number;
  testingPercentage?: number;
}): Promise<{
  trainingDataset: data.Dataset<TensorContainer>;
  validationDataset: data.Dataset<TensorContainer>;
  testingDataset: data.Dataset<TensorContainer>;
}> => {
  const samples = await extractFeatures({
    data,
    inputFeatureEngineers,
    outputFeatureEngineer
  });

  const { trainingSamples, validationSamples, testingSamples } = splitSamplesIntoTrainingValidationTestForBinaryClassification(
    samples,
    trainingPercentage,
    validationPercentage,
    testingPercentage
  );

  const trainingDataset = makeDataset(trainingSamples, batchSize);
  const validationDataset = makeDataset(validationSamples);
  const testingDataset = makeDataset(testingSamples);

  return { trainingDataset, validationDataset, testingDataset };
}
