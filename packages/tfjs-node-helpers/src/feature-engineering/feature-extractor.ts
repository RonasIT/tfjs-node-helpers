import { Feature } from './feature';

export abstract class FeatureExtractor<D, T> {
  public abstract featureType: T;

  public abstract extract(data: D): Feature<T> | Promise<Feature<T>>;
}
