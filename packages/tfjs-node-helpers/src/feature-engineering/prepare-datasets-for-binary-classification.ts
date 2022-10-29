import { data, TensorContainer } from '@tensorflow/tfjs-node';
import { splitSamplesIntoTrainingValidationTestForBinaryClassification } from '../data-splitting/training-validation-test-for-binary-classification';
import { makeDataset } from '../utils/make-dataset';
import { extractFeatures } from './extract-features';
import { FeatureExtractor } from './feature-extractor';

export const prepareDatasetsForBinaryClassification = async <D, T>({
  data,
  inputFeatureExtractors,
  outputFeatureExtractor,
  batchSize,
  trainingPercentage,
  validationPercentage,
  testingPercentage
}: {
  data: Array<D>;
  inputFeatureExtractors: Array<FeatureExtractor<D, T>>;
  outputFeatureExtractor: FeatureExtractor<D, T>;
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
    inputFeatureExtractors,
    outputFeatureExtractor
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
