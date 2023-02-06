import { Feature } from './feature';

export abstract class FeatureEngineer<T, D> {
  public abstract featureType: T;

  public abstract extractFeature(params: { data: Array<D>; index: number }): Feature<T> | Promise<Feature<T>>;
}
