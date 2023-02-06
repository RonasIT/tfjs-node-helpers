import { Feature, FeatureEngineer } from '@ronas-it/tfjs-node-helpers';
import { FeatureType } from '../enums/feature-type';
import { DatasetItem } from '../types/dataset-item';

export class GenderFeatureEngineer extends FeatureEngineer<FeatureType, DatasetItem> {
  public featureType = FeatureType.GENDER;

  public extractFeature({ data, index }: { data: Array<DatasetItem>; index: number }): Feature<FeatureType> {
    return new Feature({
      type: this.featureType,
      label: data[index].gender,
      value: (data[index].gender === 'Male') ? 1 : 0
    });
  }
}
