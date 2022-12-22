import { Feature, FeatureExtractor } from '@ronas-it/tfjs-node-helpers';
import { FeatureType } from '../enums/feature-type';
import { DatasetItem } from '../types/dataset-item';

export class AgeFeatureExtractor extends FeatureExtractor<DatasetItem, FeatureType> {
  public featureType = FeatureType.AGE;

  public extract(item: DatasetItem): Feature<FeatureType> {
    return new Feature({
      type: this.featureType,
      label: `${item.age} years`,
      value: item.age
    });
  }
}
