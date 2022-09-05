import { Feature, FeatureExtractor } from '@ronas-it/tfjs-node-helpers';
import { FeatureType } from '../enums/feature-type';
import { DatasetItem } from '../types/dataset-item';

export class GenderFeatureExtractor extends FeatureExtractor<DatasetItem, FeatureType> {
  public featureType = FeatureType.GENDER;

  public extract(item: DatasetItem): Feature<FeatureType> {
    return new Feature({
      type: this.featureType,
      label: item.gender,
      value: (item.gender === 'Male') ? 1 : 0
    });
  }
}
