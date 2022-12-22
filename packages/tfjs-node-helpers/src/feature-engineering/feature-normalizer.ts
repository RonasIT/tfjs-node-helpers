import { Feature } from './feature';

export abstract class FeatureNormalizer<T> {
  public abstract featureType: T;

  public abstract normalize(feature: Feature<T>): Feature<T> | Promise<Feature<T>>;
}
