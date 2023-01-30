import { Feature, FeatureExtractor } from '@ronas-it/tfjs-node-helpers';
import { FeatureType } from '../enums/feature-type';
import { DatasetItem } from '../types/dataset-item';

export class AnnualSalaryFeatureExtractor extends FeatureExtractor<DatasetItem, FeatureType> {
  public featureType = FeatureType.ANNUAL_SALARY;

  public extract(item: DatasetItem): Feature<FeatureType> {
    return new Feature({
      type: this.featureType,
      label: item.annual_salary.toString(),
      value: item.annual_salary
    });
  }
}
