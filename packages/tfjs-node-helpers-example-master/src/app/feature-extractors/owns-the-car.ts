import { Feature, FeatureExtractor } from '@ronas-it/tfjs-node-helpers';
import { FeatureType } from '../enums/feature-type';
import { DatasetItem } from '../types/dataset-item';

export class OwnsTheCarFeatureExtractor extends FeatureExtractor<DatasetItem, FeatureType> {
  public featureType = FeatureType.OWNS_THE_CAR;

  public extract(item: DatasetItem): Feature<FeatureType> {
    return new Feature({
      type: this.featureType,
      label: (item.owns_the_car === 0) ? 'Not purchased' : 'Purchased',
      value: item.owns_the_car
    });
  }
}
