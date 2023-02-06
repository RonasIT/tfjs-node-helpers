import { Feature, MinMaxNormalizedFeatureEngineer } from '@ronas-it/tfjs-node-helpers';
import { FeatureType } from '../enums/feature-type';
import { DatasetItem } from '../types/dataset-item';

export class AnnualSalaryFeatureEngineer extends MinMaxNormalizedFeatureEngineer<FeatureType, DatasetItem> {
  public featureType = FeatureType.ANNUAL_SALARY;

  public extractFeature({
    data,
    index
  }: {
    data: Array<DatasetItem>;
    index: number;
  }): Feature<FeatureType> {
    const feature = new Feature({
      type: this.featureType,
      label: data[index].annual_salary.toString(),
      value: data[index].annual_salary
    });

    return this.normalizeFeature({
      feature,
      featureValues: data.map((item) => item.annual_salary)
    });
  }
}
