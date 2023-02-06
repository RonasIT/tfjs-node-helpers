import { Feature, MinMaxNormalizedFeatureEngineer } from '@ronas-it/tfjs-node-helpers';
import { FeatureType } from '../enums/feature-type';
import { DatasetItem } from '../types/dataset-item';

export class AgeFeatureEngineer extends MinMaxNormalizedFeatureEngineer<FeatureType, DatasetItem> {
  public featureType = FeatureType.AGE;

  public extractFeature({
    data,
    index
  }: {
    data: Array<DatasetItem>;
    index: number;
  }): Feature<FeatureType> {
    const feature = new Feature({
      type: this.featureType,
      label: `${data[index].age} years`,
      value: data[index].age
    });

    return this.normalizeFeature({
      feature,
      featureValues: data.map((item) => item.age)
    });
  }
}
