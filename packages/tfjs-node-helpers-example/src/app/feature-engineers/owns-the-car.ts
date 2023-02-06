import { Feature, FeatureEngineer } from '@ronas-it/tfjs-node-helpers';
import { FeatureType } from '../enums/feature-type';
import { DatasetItem } from '../types/dataset-item';

export class OwnsTheCarFeatureEngineer extends FeatureEngineer<FeatureType, DatasetItem> {
  public featureType = FeatureType.OWNS_THE_CAR;

  public extractFeature({ data, index }: { data: Array<DatasetItem>; index: number }): Feature<FeatureType> {
    return new Feature({
      type: this.featureType,
      label: (data[index].owns_the_car === 0) ? 'Not purchased' : 'Purchased',
      value: data[index].owns_the_car
    });
  }
}
