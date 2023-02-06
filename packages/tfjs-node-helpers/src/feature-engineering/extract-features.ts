import { FeatureEngineer } from './feature-engineer';
import { Sample } from '../training/sample';

export const extractFeatures = async <D, T>({
  data,
  inputFeatureEngineers,
  outputFeatureEngineer
}: {
  data: Array<D>;
  inputFeatureEngineers: Array<FeatureEngineer<T, D>>;
  outputFeatureEngineer: FeatureEngineer<T, D>;
}): Promise<Array<Sample>> => {
  const samples = [];

  for (let i = 0; i < data.length; i++) {
    const [inputFeatures, outputFeature] = await Promise.all([
      Promise.all(
        inputFeatureEngineers.map((engineer) => engineer.extractFeature({ data, index: i }))
      ),
      outputFeatureEngineer.extractFeature({ data, index: i })
    ]);

    const input = inputFeatures.map((feature) => feature.value);
    const output = [outputFeature.value];

    samples.push({ input, output });
  }

  return samples;
};
